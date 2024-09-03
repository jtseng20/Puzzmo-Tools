import torch
import torch.nn.functional as F

import random

def extract_and_replace_vectorized_torch(tensor, wild_card, num_sims):
    """Extracts rows containing a wild card and replaces with a selected range using vectorization."""
    # batch x row x rank
    # Find indices of rows containing wild card
    indices = torch.where(torch.any(tensor == wild_card, dim=-1))

    # Extract those rows
    extracted_rows = tensor[indices]
    extracted_rows = extracted_rows.unsqueeze(1).repeat(1, num_sims, 1)

    # Create a replacement tensor with numbers 0-n
    replacement_tensor = torch.arange(num_sims).unsqueeze(0).repeat(len(extracted_rows), 1).unsqueeze(-1).to(tensor.device)

    # Replace -2 with the replacement tensor
    new_tensor = torch.where(extracted_rows == wild_card, replacement_tensor, extracted_rows)

    return new_tensor, indices

def generate_nmodal_distribution(batch_size, num_categories):
    num_centers = torch.randint(1, 3, (1,))
    centers = torch.rand((batch_size, num_centers, 1)) * num_categories
    scales = torch.rand((batch_size, num_centers, 1)) * 100 + 60

    i = torch.tile(torch.tile(torch.arange(num_categories), (num_centers, 1)), (batch_size, 1, 1))
    diff = torch.exp(-((i - centers) ** 2) / scales)
    dist = diff.sum(1) + torch.ones(batch_size, num_categories) / 5
    dist = dist / dist.sum(-1, keepdim=True)

    return dist

def p_2p_3_4(tensor):
    # Expand tensor for broadcasting
    tensor_expanded = tensor[:, :, None, :]  # Shape: (b, c, 1, 4)
    
    # Create a comparison tensor to find matches
    matches = (tensor_expanded == tensor_expanded.transpose(-2, -1)) & (tensor_expanded != NO_CARD)  # Shape: (b, c, 4, 4)
    mask = ~torch.eye(matches.shape[-1], dtype = matches.dtype, device = matches.device)[None, None]
    matches &= mask
    
    # Sum along the last axis to count occurrences of each number
    counts = matches.sum(dim=-1)  # Shape: (b, 4)
    return ((counts.sum(dim=-1)))

def flush(tensor):
    # Check if all elements in each row are the same
    # This will return a boolean tensor where True means all elements in that row are identical
    identical = (tensor == tensor[:, :, 0:1]).all(dim=-1)
    is_nonzero = (tensor != NO_CARD).all(dim=-1)
    return identical & is_nonzero

def check_straights(tensor):
    # Sort each row and check if the difference between consecutive elements is 1
    ace_replace = torch.where(tensor == 0, 13, tensor) # replace 0s with 13s to check for both ace high and ace low straights
    
    sorted_tensor = torch.sort(tensor, dim=-1).values
    differences = sorted_tensor[:, :, 1:] - sorted_tensor[:, :, :-1]  # Calculate differences
    is_straight = (differences == 1).all(dim=-1)  # Check if all differences are 1

    sorted_tensor_2 = torch.sort(ace_replace, dim=-1).values
    differences_2 = sorted_tensor_2[:, :, 1:] - sorted_tensor_2[:, :, :-1]  # Calculate differences
    is_straight_2 = (differences_2 == 1).all(dim=-1)  # Check if all differences are 1
    
    is_nonzero = (tensor != NO_CARD).all(dim=-1)
    return (is_straight | is_straight_2) & is_nonzero

rank_to_idx = {"A":0, "K": 1, "Q": 2, "J": 3, "10": 4, "9": 5, "8": 6, "7": 7, "6": 8, "5": 9, "4": 10, "3": 11, "2": 12, "W": 14}
idx_to_rank = {idx:rank for rank, idx in rank_to_idx.items()}
WILD_RANK = rank_to_idx["W"]

suit_to_idx = {"S": 0, "H": 1, "D": 2, "C": 3, "W": 4}
idx_to_suit = {idx:suit for suit, idx in suit_to_idx.items()}
WILD_SUIT = suit_to_idx["W"]

NO_CARD = -1

def mask_policy(policy, board_dim, occ, has_discarded, cards_placed, non_discard_slots_mask, discard_slots_mask):
    # mask out illegal moves
    policy = policy * occ
    # now mask out the discard slots for every row that has already discarded
    policy[has_discarded] *= non_discard_slots_mask
    # if it's the last card of the hand, force a discard if necessary
    policy[~has_discarded & (cards_placed == board_dim)] *= discard_slots_mask
    # enforce unit norm
    policy = F.normalize(policy, p=1, dim=-1)
    return policy

class PileUpPokerProEnv:
    def __init__(self, batch_size, board_dim = 4, mode = "auto", device="cpu"):
        assert board_dim in (4,5)
        self.batch_size = batch_size
        self.action_dim = action_dim = board_dim * (board_dim + 1)
        self.mode = mode
        self.board_dim = board_dim
        self.num_ranks = 9 if board_dim == 4 else 13
        self.r = torch.arange(batch_size)
        self.row_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, board_dim + 1)
        self.col_indices = torch.arange(board_dim+1).unsqueeze(0).expand(batch_size, board_dim + 1)
        # utility mask for discard pile slots
        self.discard_slots_mask = torch.zeros((1, action_dim), device=device)
        self.discard_slots_mask[:, -board_dim:] = 1

        self.non_discard_slots_mask = torch.zeros((1, action_dim), device=device)
        self.non_discard_slots_mask[:, :-board_dim] = 1

        self.score_dict = torch.tensor([  0., 80., 5., 80., 60., 0., 125., 0., 230., 0., 0., 0.,
                                325., 0., 0., 0., 0., 0., 0., 0., 180., 450.], device=device)

        self.device = device
        
        self.reset(mode == "manual")  # Initialize the game

    def to(self, device):
        self.device = device
        # Iterate over the attributes of the class
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                # Move the tensor to the specified device
                setattr(self, attr_name, attr_value.to(device))

    def reset(self, ignore_input=False):
        """
        Reset the game state to the initial configuration.
        This includes setting up the grid, the hand, and resetting the game status.
        Returns:
            observation: The initial state of the game.
        """
        board_dim = self.board_dim
        self.state = torch.zeros(self.batch_size, board_dim + 1, board_dim + 1, self.num_ranks + 4 + (2 if self.board_dim == 5 else 0), device = self.device)
        self.rank = NO_CARD * torch.ones(self.batch_size, board_dim + 1, board_dim, dtype=torch.int64, device = self.device)
        self.suit = NO_CARD * torch.ones(self.batch_size, board_dim + 1, board_dim, dtype=torch.int64, device = self.device)
        self.cardbank = torch.ones(self.batch_size, self.num_ranks * 4 + (5 if self.board_dim == 5 else 0), device = self.device) # wild card for 5x5
        
        # sometimes make the distribution more interesting
        mask = torch.rand(self.batch_size) < 0.5
        nmodal = generate_nmodal_distribution(self.batch_size, self.num_ranks * 4 + (5 if self.board_dim == 5 else 0)).to(self.device)
        self.cardbank[mask] = nmodal[mask]

        # mask out indices 52 - 56 (wild card is 56, the other indices aren't real)
        if self.board_dim == 5:
            self.cardbank[:, 52:-1] = 0
        
        self.done = False  # Reset the game over status
        self.draw_hand(ignore_input)
        self.turns_done = 0
        # occupancy mask for move legality masking 
        self.occ = torch.ones(self.batch_size, board_dim * (board_dim + 1), device = self.device)
        
        # game starts with a random hand
        return self._get_observation()

    def draw_hand(self, ignore_input):
        if self.mode == "auto" or ignore_input:
            selections = torch.multinomial(self.cardbank, self.board_dim + 1)
        else:
            # manual input for live game testing
            assert self.batch_size == 1, "Manual input does not support batch_size > 1"
            handstring = ""
            while len(handstring.split()) != (self.board_dim + 1):
                handstring = input(f"Enter {self.board_dim + 1} cards for next hand: ").upper()

            selections = torch.zeros(1, self.board_dim + 1, dtype=torch.int64, device=self.device)
            for num, card in enumerate(handstring.split()):
                rank, suit = card[:-1], card[-1]
                rank, suit = rank_to_idx[rank], suit_to_idx[suit]
                selections[0, num] = (rank * 4 + (suit % 4))
        self.cardbank[self.row_indices, selections] = 0 # cannot be picked anymore
        # write selections into the state
        rank = selections // 4
        suit = selections % 4
        self.state[self.row_indices, self.col_indices, -1, rank] = 1
        self.state[self.row_indices, self.col_indices, -1, -suit-1] = 1
        self.cards_placed = 0
        self.has_discarded = torch.zeros(self.batch_size, dtype=bool, device=self.device) # for legal move masking
        self.hand = selections

    def _get_observation(self):
        """
        Returns the current state of the game as the observation.
        This combines the grid and the hand into a single representation.
        """
        return self.state

    def step(self, action):
        """
        Takes an action in the environment and updates the game state accordingly.
        Args:
            action: The action to take, batch x 1 (card position)
        
        Returns:
            observation: The updated game state.
            reward: The reward obtained after taking the action.
            done: Whether the game has ended.
            info: Additional debugging information, if any.
        """
        if self.done:
            raise ValueError("Game is over. Please reset the environment.")

        # place down one card
        r = self.r
        place_idx = action # the action contains only the place to put the card
        card_idx = self.hand[:, self.cards_placed] # because the agent always places the cards in its hand from left to right
        
        x, y = place_idx // self.board_dim, place_idx % self.board_dim
        rank, suit = card_idx // 4, card_idx % 4
        self.rank[r, x, y] = rank
        self.suit[r, x, y] = suit
        # wild card indexing is dumb
        self.suit = torch.where(self.rank == WILD_RANK, torch.tensor(WILD_SUIT, dtype=self.suit.dtype), self.suit)
        self.state[r, x, y, rank] = 1
        self.state[r, x, y, -suit-1] = 1

        # clear the placed card from the hand column
        self.state[:, self.cards_placed, -1] = 0
        # update occupancy
        # sanity check
        assert (self.occ[self.r, place_idx] != 0).all()
        self.occ[self.r, place_idx] = 0

        # update discard status
        is_discard = (self.board_dim ** 2 <= place_idx)
        self.has_discarded |= is_discard

        self.cards_placed += 1

        # on manual mode, render before drawing new hand
        if self.mode == "manual":
            self.render()
        # if hand is empty, increments turn number and draw a new hand
        if self.cards_placed == self.board_dim + 1:
            self.turns_done += 1
            if self.turns_done < self.board_dim:
                self.draw_hand(False)
        
        # Check if the game is done
        self.done = self._check_game_over()
        
        # Calculate reward based on the new game state
        reward = self._calculate_reward()
        
        # Get the next observation
        observation = self._get_observation()
        
        # Placeholder for additional debugging information
        info = {}
        
        return observation, reward, torch.full_like(reward, self.done), info

    def _check_game_over(self):
        """
        Check if the game has ended.
        Returns:
            done: A boolean indicating whether the game has ended.
        """
        done = (self.turns_done == self.board_dim)
        # sanity checks
        if done:
            assert self.occ.sum() == 0
        return done

    def make_board(self, x):
        col = x[:, :-1].transpose(-1, -2)
        corner_x = [0, -2, 0, -2] + ([2] if self.board_dim == 5 else [])
        corner_y = [0, 0, -1, -1] + ([2] if self.board_dim == 5 else [])
        corner = x[:, corner_x, corner_y]
        return torch.cat((corner[:, None], col, x), dim=1)
    
    def _calculate_reward(self):
        """
        Calculate the reward based on the current state of the grid.
        Returns:
            reward: The computed reward for the current state.
        """
        rankboard = self.make_board(self.rank)
        suitboard = self.make_board(self.suit)
        # 1 row corners
        # n rows of columns
        # n rows of rows
        # 1 row discard

        # in advance of calculating true scores, extract any row with a wild card 
        # and simulate every single possible value for said wild card, separately by rank and suit
        wild_card_ranks, r_idx = extract_and_replace_vectorized_torch(rankboard, WILD_RANK, self.num_ranks)
        wild_card_suits, s_idx = extract_and_replace_vectorized_torch(suitboard, WILD_SUIT, 4)

        possible_straights = check_straights(wild_card_ranks).max(-1)[0] * 20
        possible_combos = p_2p_3_4(wild_card_ranks).max(-1)[0]
        possible_flushes = flush(wild_card_suits).max(-1)[0] # the only possible flushes + rank combos are flush+pair and flush+straight
        sim_scores = torch.max(possible_straights, possible_combos) + possible_flushes
        
        pairscore = p_2p_3_4(rankboard)
        flushscore = flush(suitboard)
        straightscore = check_straights(rankboard)
        # replace existing scores with wildcard maxes. By construction, the wildcard scores must be greater than scores without them
        scorevec = 20 * straightscore + pairscore + flushscore
        
        if len(sim_scores):
            scorevec[r_idx] = sim_scores
        scorevec[:, -1] *= ~(scorevec[:, :-1] == 0).any(dim=-1)
        scorevec = self.score_dict[scorevec]
        # score mutipliers (2x corners, 3x discard)
        scorevec[:, 0] *= 2
        scorevec[:, -1] *= 3
        # multipliers for number of hands
        num_hands = torch.count_nonzero(scorevec, dim=-1) // 2 + (self.board_dim < 5)
        scorevec *= torch.clamp(num_hands[:, None], 1, None) # no zero multipliers lol
        return scorevec.sum(dim=-1) # b,

    def legalize_policy(self, policy):
        return mask_policy(policy, self.board_dim, self.occ, 
                           self.has_discarded, self.cards_placed, 
                           self.non_discard_slots_mask, self.discard_slots_mask)

    def pick_move(self, policy):
        policy = self.legalize_policy(policy)
        return policy.argmax(dim=-1)

    def stringify(self, idx):
        idx = idx.item()
        if idx != NO_CARD * 5:
            rank, suit = (idx // 4, idx % 4) if idx != (WILD_RANK * 4 + WILD_SUIT) else (WILD_RANK, WILD_SUIT)
            rankname = idx_to_rank[rank]
            suitname = idx_to_suit[suit]
        else:
            return "  "
        return rankname+suitname
    
    def render(self):
        """
        Render the current game state for debugging purposes.
        """
        print("Board:")
        print("=====" * self.board_dim + "=")
        for i in range(self.board_dim):
            s = ""
            for j in range(self.board_dim):
                s += f"| {self.stringify(self.rank[0, i, j] * 4 + self.suit[0, i, j])} "
            print(s + "|")
            print("=====" * self.board_dim + "=")
        print()
        print("Hand:")
        print("=====" * (self.board_dim+1) + "==")
        s = ""
        for j in range(self.board_dim + 1):
            s += f"| {self.stringify(self.hand[0, j]) if j >= self.cards_placed else '  '} "
        print(s + "|")
        print("=====" * (self.board_dim+1) + "==")
        print()
        print("Discard:")
        print("=====" * self.board_dim + "=")
        s = ""
        for j in range(self.board_dim):
            s += f"| {self.stringify(self.rank[0, -1, j] * 4 + self.suit[0, -1, j])} "
        print(s + "|")
        print("=====" * self.board_dim + "=")
        print()