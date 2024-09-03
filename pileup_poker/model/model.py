import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import StatTracker
import math

class ResModulev0(nn.Module):
    def __init__(self, dim):
        super(ResModule, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x + residual
        
class ResModulev1(nn.Module):
    def __init__(self, dim):
        super(ResModule, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = F.mish

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.act(x + residual)

class ResModulev2(nn.Module):
    def __init__(self, dim, board_dim):
        super(ResModulev2, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([dim, board_dim, board_dim])  # Adjust the dimensions here as needed
        
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([dim, board_dim, board_dim])  # Adjust the dimensions here as needed
        
        self.act = F.mish

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.ln2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual

class PopArt(nn.Module):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            beta: float = 0.5
    ):
        super().__init__()

        self.beta = beta
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        
        self.register_buffer('mu', torch.zeros(output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_features, requires_grad=False))
        self.register_buffer('v', torch.ones(output_features, requires_grad=False))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        normalized_output = x.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)

        return normalized_output

    def update(self, value):
        self.mu = self.mu.to(value.device)
        self.sigma = self.sigma.to(value.device)
        self.v = self.v.to(value.device)
        
        old_mu = self.mu
        old_std = self.sigma
        batch_mean = torch.mean(value, 0)
        batch_v = torch.mean(torch.pow(value, 2), 0)
        batch_mean = torch.where(torch.isnan(batch_mean), self.mu, batch_mean)
        batch_v = torch.where(torch.isnan(batch_v), self.v, batch_v)

        # EMA
        batch_mean = (1 - self.beta) * self.mu + self.beta * batch_mean
        batch_v = (1 - self.beta) * self.v + self.beta * batch_v
        batch_std = torch.sqrt(batch_v - (batch_mean ** 2))
        # Clip the standard deviation to reject the outlier data.
        batch_std = torch.clamp(batch_std, min=1e-4, max=1e+6)
        # Replace the nan values
        batch_std = torch.where(torch.isnan(batch_std), self.sigma, batch_std)

        # Update the normalization parameters.
        self.mu = batch_mean
        self.v = batch_v
        self.sigma = batch_std
        
        self.weight.data = (self.weight.t() * old_std / self.sigma).t()
        self.bias.data = (old_std * self.bias + old_mu - self.mu) / self.sigma

class MyCNN(nn.Module):
    def __init__(self, board_dim=4, dim=64):
        super().__init__()
        
        num_ranks = 9 if board_dim == 4 else 15
        self.conv1 = nn.Conv2d(num_ranks + 4 + 2, dim, kernel_size=3, padding=1) # ranks + 4 suits + rank embed + suit embed
        self.bn1 = nn.BatchNorm2d(dim)
        
        res_modules = []
        for _ in range(board_dim):
            res_modules.append(ResModulev2(dim, board_dim + 1))
        self.res_modules = nn.Sequential(*res_modules)

        # Policy head
        self.head1 = nn.Conv2d(dim, 2, kernel_size=1)
        self.bn_head1 = nn.BatchNorm2d(2)
        self.head1_fc = nn.Linear(2 * (board_dim + 1) * (board_dim + 1), 128)
        self.head1_out = nn.Linear(128, (board_dim + 1) * board_dim)  # Output for n * (n+1) actions
        self.pi_dropout = nn.Dropout(p=0.1)
        
        # Value head
        self.v_res = ResModulev2(dim, board_dim + 1)
        self.head2 = nn.Conv2d(dim, 4, kernel_size=1)
        self.bn_head2 = nn.BatchNorm2d(4)
        self.head2_fc = nn.Linear(4 * (board_dim + 1) * (board_dim + 1), 128)
        #self.head2_out = nn.Linear(128, 1) # if not using POP-ART adaptive return normalization
        self.head2_out = PopArt(128, 1)
        self.val_dropout = nn.Dropout(p=0.1)

        self.rank_embed = nn.Embedding(num_ranks + 1 + (board_dim == 5), 16)
        self.rank_proj = nn.Linear(16, (board_dim + 1) * (board_dim + 1))

        self.suit_embed = nn.Embedding(4, 16)
        self.suit_proj = nn.Linear(16, (board_dim + 1) * (board_dim + 1))
        self.board_dim = board_dim

        self.act = F.mish
        
    def forward(self, x, idx, temperature=1.):
        b = x.shape[0]
        rank, suit = idx // 4, idx % 4
        dim = (self.board_dim + 1)
        r_embed = self.rank_proj(self.rank_embed(rank)) # b x 5 x 25
        r_embed = r_embed.reshape(b, -1, dim, dim)
        s_embed = self.suit_proj(self.suit_embed(suit)) # b x 5 x 25
        s_embed = s_embed.reshape(b, -1, dim, dim)
        # Input x shape: (b, 5, 5, 13)
        x = x.permute(0, 3, 1, 2)  # Change x to shape (b, 13, 5, 5)
        # add on the embedding
        x = torch.cat((x, r_embed, s_embed), dim=1) # b, 23, 5, 5
        x = self.act(self.bn1(self.conv1(x)))
        x = self.res_modules(x)

        # Policy head
        x1 = self.act(self.bn_head1(self.head1(x)))
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.pi_dropout(self.act(self.head1_fc(x1)))
        policy = torch.clamp(self.head1_out(x1) / temperature, -20, 20)
        policy = F.softmax(policy, dim=-1)

        # Value head
        x2 = self.act(self.bn_head2(self.head2(self.v_res(x))))
        x2 = x2.reshape(x2.size(0), -1)
        value = self.head2_out(self.val_dropout(self.act(self.head2_fc(x2)))).squeeze(-1)
        
        return policy, value