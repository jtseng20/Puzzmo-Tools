import os
from pathlib import Path

import torch
from torch.nn import functional as F
from model.adan import Adan
from torch.utils.data import DataLoader
from collections import deque
from model.model import MyCNN
from model.utils import SimpleDataset, RollingBuffer, to_device, explained_variance, CosineScheduler
from game.pro_env import PileUpPokerProEnv, mask_policy
from data_utils import increment_path, unflatten, reflatten

from tqdm import tqdm
import wandb
import pickle

# Hyperparameters
gamma = 0.95  # Discount factor
lambda_gae = 0.95  # Lambda for GAE
clip_param = 0.2  # PPO clipping parameter
learning_rate = 2e-4

NO_CARD = 36 # sentinel

def compute_gae(rewards, values, masks, gamma=gamma, lambda_=lambda_gae):
    """
    Compute Generalized Advantage Estimation (GAE).
    """
    next_value = torch.zeros(values.shape[0], 1).to(values.device)
    values = torch.cat([values, next_value], dim=-1)
    gae = 0
    returns = deque()
    for step in reversed(range(rewards.shape[-1])):
        delta = rewards[:, step] + gamma * values[:, step + 1] * masks[:, step] - values[:, step]
        gae = delta + gamma * lambda_ * masks[:, step] * gae
        returns.appendleft(gae + values[:, step])
    return torch.stack(list(returns)).T

def collect_trajectory(env, model, max_steps, alpha = 0.3, eps = 0.75, temp = 1.):
    """
    Collects a trajectory by interacting with the environment.
    """
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    masks = []
    terminal_values = []
    cards = []
    # state information for legalizing policy
    occs = []
    discards = []
    cards_placed = []

    alpha = torch.full((env.batch_size, env.board_dim * (env.board_dim + 1)), alpha).to(env.device)
    dirichlet = torch.distributions.Dirichlet(alpha)
    
    state = env.reset()
    model.to(env.device)
    model.eval()

    print(f"Generating Data for {max_steps} Timesteps")
    for _ in tqdm(range(max_steps)):
        card_idxs = env.hand[:, env.cards_placed]
        #card_idxs[:, :env.cards_placed] = NO_CARD
        with torch.no_grad():
            # Get policy and value from the model
            policy, value = model(state, card_idxs, temp)
            # legalize policy and add action noise
            noisy_policy = policy * eps + dirichlet.sample() * (1 - eps)
            policy = env.legalize_policy(policy)
            noisy_policy = env.legalize_policy(noisy_policy)

            # Sample action from the policy distribution
            dist = torch.distributions.Categorical(probs=noisy_policy)
            action = dist.sample()

            # Get log probability of the action
            log_prob = dist.log_prob(action)

        # Store the trajectory data (batched)
        states.append(state.cpu())
        actions.append(action.cpu())
        log_probs.append(log_prob.cpu())
        values.append(value.cpu())
        cards.append(card_idxs.cpu())
        occs.append(env.occ.cpu())
        discards.append(env.has_discarded.cpu())
        cards_placed.append(torch.full_like(card_idxs, env.cards_placed).cpu())

        next_state, reward, done, _ = env.step(action)
        rewards.append(reward.cpu())
        masks.append((1 - done).cpu())
        state = next_state

        if env.done:
            # at the terminal state, add a new batch of terminal states. Value doesn't really matter since GAE never considers them...
            next_value = torch.zeros_like(value).cpu()
            terminal_values.append(next_value.cpu())

            # and then reset
            state = env.reset()

    # cat all the returns together
    states = torch.cat(states, dim=0)
    actions = torch.cat(actions, dim=0)
    rewards = torch.stack(rewards, dim=0)
    log_probs = torch.cat(log_probs, dim=0)
    values = torch.stack(values, dim=0)
    masks = torch.stack(masks, dim=0)
    cards = torch.cat(cards, dim=0)

    occs = torch.cat(occs, dim=0)
    discards = torch.cat(discards, dim=0)
    cards_placed = torch.cat(cards_placed, dim=0)

    assert max_steps % env.action_dim == 0
    n = max_steps // env.action_dim
    assert len(terminal_values) == n
    # reshape rewards, values, masks, next_value for GAE to (batch x timesteps)
    rewards = unflatten(rewards, n).T
    values = unflatten(values, n).T
    masks = unflatten(masks, n).T
    terminal_values = torch.cat(terminal_values)[:, None]

    print(f"Generated {len(states)} Samples")

    return states, actions, rewards, log_probs, values, masks, terminal_values, cards, occs, discards, cards_placed

class PPO:
    def __init__(self, gen_batch_size=10000, max_gen_steps=2048, mini_batch_size=64, num_epochs=10, buffer_len=None, dirichlet_eps=0.75, device="cpu", wandb_log=True):
        self.board_dim = board_dim = 5
        self.env = PileUpPokerProEnv(gen_batch_size, board_dim=board_dim, device=device)
        self.model = MyCNN(board_dim, 128).to(device)
        
        self.optim = Adan(self.model.parameters(), lr=learning_rate)
        self.batch_size = mini_batch_size
        self.device = device
        self.ppo_epochs = num_epochs
        self.max_gen_steps = max_gen_steps
        self.wandb_log = wandb_log
        assert max_gen_steps % self.env.action_dim == 0
        self.n = max_gen_steps // self.env.action_dim
        self.buffer_len = buffer_len
        self.dirichlet_eps = dirichlet_eps

        self.dataset = None
        self.dataloader = None

    def update(self, states, actions, rewards, log_probs, values, masks, cards, occs, discards, cards_placed, temperature=1.):
        """
        Perform PPO update using collected trajectories.
        """
        # report final return (final step reward)
        final_scores = rewards[:, -1].clone()
        # convert rewards to stepwise rewards
        rewards[:, 1:] = rewards[:, 1:] - rewards[:, :-1]

        mu, sigma = self.model.head2_out.mu.cpu(), self.model.head2_out.sigma.cpu()
        unnormalized_values = values * sigma + mu
        returns = compute_gae(rewards, unnormalized_values, masks)
        advantages = returns - unnormalized_values
    
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5) # technically this could be more accurate wrt the rolling buffer

        # flatten advantages, returns
        advantages = reflatten(advantages.T, self.n)
        returns = reflatten(returns.T, self.n)

        if self.dataset is None:
            self.dataset = RollingBuffer(states, actions, log_probs, 
                                         returns, advantages, cards, occs, 
                                         discards, cards_placed, max_len=self.buffer_len)
            
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
            )
        else:
            # if the dataset already exists, push new data into the buffer
            print(f"Extending Data Buffer by {len(states)} items")
            self.dataset.push(states, actions, log_probs, returns, advantages, cards, occs, discards, cards_placed)
            print(f"Data Buffer now contains {len(self.dataset)} items")
        
        self.model.to(self.device)
        self.model.train()

        print(f"Updating for {self.ppo_epochs} epochs")
        for _ in range(self.ppo_epochs):
            for batch in tqdm(self.dataloader):
                (sampled_states, sampled_actions, sampled_log_probs, 
                    sampled_returns, sampled_advantages, card_idx, occ, 
                    discard, cards_) = to_device(batch, self.device)
                
                # Get new log probabilities and values
                new_policy, new_values = self.model(sampled_states, card_idx, temperature)
                # legalize policy
                new_policy = mask_policy(new_policy, self.board_dim, 
                                         occ, discard, cards_, 
                                         self.env.non_discard_slots_mask, self.env.discard_slots_mask)
                new_dist = torch.distributions.Categorical(probs=new_policy)
                new_log_probs = new_dist.log_prob(sampled_actions)
                entropy = new_dist.entropy().mean()
    
                # Calculate ratio for policy update
                ratio = (new_log_probs - sampled_log_probs).exp()
                clip_fraction = (ratio > (1 + clip_param)).float().mean() + (ratio < (1 - clip_param)).float().mean()
    
                # Clipped policy objective
                surr1 = ratio * sampled_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * sampled_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
    
                # Value loss (Mean Squared Error)

                # normalize returns with popart stats
                mu, sigma = self.model.head2_out.mu, self.model.head2_out.sigma
                normalized_returns = (sampled_returns - mu) / sigma
                value_loss = F.mse_loss(normalized_returns, new_values) # values from the model are in normalized space
                # explained variance
                unnormalized_values = new_values * sigma + mu
                exp_variance = explained_variance(unnormalized_values, sampled_returns)
    
                # Total loss (combine policy and value loss)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    
                # Optimize the model
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # update stats for popart
                self.model.head2_out.update(sampled_returns)
                
                # Log metrics to WandB (if enabled)
                if self.wandb_log:
                    wandb.log({
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "total_loss": loss.item(),
                        "average_reward": final_scores.mean().item(),
                        "entropy": entropy.item(),
                        "clip_fraction": clip_fraction.item(),
                        "value_explained_variance": exp_variance.item(),
                        "return_mean": mu.item(),
                        "return_sigma": sigma.item(),
                        "value_mean": unnormalized_values.mean().item(),
                        "value_sigma": unnormalized_values.std().item()
                    })


    def train_loop(self, num_iters=1000):
        save_dir = str(increment_path(Path("runs/train") / "exp"))
        exp_name = save_dir.split("/")[-1]

        if self.wandb_log:
            wandb.init(project="PUP", name=f"{exp_name}")
        save_dir = Path(save_dir)
        wdir = save_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)

        eps_scheduler = CosineScheduler(1., 1., 100) # mixing weight for dirichlet noise (currently unused)
        alpha_scheduler = CosineScheduler(0.7, 0.7, 500) # spread parameter for dirichlet noise (currently unused)
        temp_scheduler = CosineScheduler(0.5, 1, 10) # softmax temperature
        
        for update in range(1, num_iters + 1):
            print(f"Starting iteration {update}")
            # Collect trajectory
            alpha = alpha_scheduler.current_value()
            eps = eps_scheduler.current_value()
            tau = 1 / temp_scheduler.current_value()
            
            (states, actions, rewards, log_probs, 
                 values, masks, next_value, cards, 
                 occs, discards, cards_placed) = collect_trajectory(self.env, self.model, 
                                                                self.max_gen_steps, 
                                                                alpha, eps, tau)
            # Perform PPO update
            self.update(states, actions, rewards, log_probs, 
                            values, masks, cards, occs, 
                            discards, cards_placed, tau)
            
            if (update % 10) == 0:
                # save model
                ckpt = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                }
                save_path = os.path.join(wdir, f"train-{update}.pt")
                torch.save(ckpt, save_path)
                print(f"[MODEL SAVED at Iteration {update}: {save_path}]")
            # update dirichlet mix eps
            eps_scheduler.step()
            alpha_scheduler.step()
            temp_scheduler.step()
        print("Training done")