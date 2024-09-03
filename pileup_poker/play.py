import torch
from torch.nn import functional as F
from model.model import MyCNN
from game.pro_env import PileUpPokerProEnv, mask_policy

class TestRig:
    def __init__(self, board_dim=4, device="cpu", mode="manual"):
        self.board_dim = board_dim
        self.env = PileUpPokerProEnv(1, board_dim=board_dim, mode=mode, device=device)
        self.model = MyCNN(board_dim, 128).to(device)
        
        checkpoint = torch.load(
                f"best{board_dim}x{board_dim}.pt", map_location=device
            )
        self.model.load_state_dict(
                        checkpoint["model_state_dict"]
                    )
    def test(self):
        # start a new game
        env = self.env
        model = self.model
        state = env.reset()
        model.to(env.device)
        model.eval()
        mu, sigma = model.head2_out.mu, model.head2_out.sigma
        
        while True:
            env.render()
            card_idxs = env.hand[:, env.cards_placed]
            with torch.no_grad():
                # Get policy and value from the model
                policy, value = model(state, card_idxs)
                policy = env.legalize_policy(policy)
    
                # Sample action from the policy distribution
                action = policy.argmax(dim=-1)
                idx = action.item()
                if idx < self.board_dim ** 2:
                    print(f"Model places {env.stringify(card_idxs[0])} at {idx // self.board_dim, idx % self.board_dim} with certainty {(policy[0, action].item() * 100):.2f}%")
                else:
                    print(f"Model discards {env.stringify(card_idxs[0])} with certainty {(policy[0, action].item() * 100):.2f}%")
            print()
            print()
            next_state, reward, done, _ = env.step(action)
            print(f"Estimated Value: {(value * sigma + mu + reward).item():.2f}")
            state = next_state
    
            if env.done:
                env.render()
                print(f"Final Reward: {env._calculate_reward().item()}")
                break

if __name__ == "__main__":
    while True:
        dim = input("Play 4x4 or 5x5? [4, 5]: ")
        if dim in ["4", "5"]:
            dim = int(dim)
            break
        else:
            print("Invalid input; please enter [4] or [5]")
    rig = TestRig(board_dim=dim, device="cpu", mode="manual") # manual = enter the cards yourself, auto = randomly play out a game
    rig.test()
