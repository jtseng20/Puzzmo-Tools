import random
import math

import torch
from torch.utils.data import Dataset, DataLoader
from collections import deque

def to_device(data, device):
    return (x.to(device) if isinstance(x, torch.Tensor) else x for x in data)

class HackLoader:
    def __init__(self, dataset, batch_size=1, drop_last=True, shuffle=True, device="cpu"):
        self.dataset = dataset
        self.shuffle = shuffle
        self.device = device
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(dataset)
        self.indices = list(range(self.num_samples))
        self.current_index = 0
        self.reset()

    def collate_fn(self, x):
        # x is a tuple of data, we need to collate them in a type-aware way
        if isinstance(x[0], torch.Tensor):
            return torch.stack(x).to(self.device)
        else:
            return x

    def __iter__(self):
        self.current_index = 0
        return self

    def __len__(self):
        func = math.floor if self.drop_last else math.ceil
        return func(len(self.dataset) / self.batch_size)

    def __next__(self):
        if self.current_index > self.num_samples - (self.batch_size if self.drop_last else 1):
            self.reset()  # Reset the iterator automatically
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        if isinstance(batch[0], tuple):
            # if the output is a tuple, transpose it while collating
            batch = tuple(map(self.collate_fn, zip(*batch)))
        else:
            # collate if the output is just a single item
            batch = self.collate_fn(batch)

        self.current_index += self.batch_size

        return batch

    # For Python 2 compatibility
    next = __next__

    def reset(self):
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.indices)  # Shuffle indices on reset  

class SimpleDataset(Dataset):
    def __init__(self, *lists):
        self.lists = lists
        assert len(set([len(x) for x in lists])) == 1, "Every input must be of the same length"

    def __getitem__(self, index):
        items = [lst[index] for lst in self.lists]
        return tuple(items) if len(items) > 1 else items[0]

    def __len__(self):
        return len(self.lists[0])

class RollingBuffer(Dataset):
    def __init__(self, *lists, max_len=None):
        self.lists = list(lists)
        self.max_len = int(max_len)
        assert len(set([len(x) for x in lists])) == 1, "Every input must be of the same length"

    def __getitem__(self, index):
        items = [lst[index] for lst in self.lists]
        return tuple(items) if len(items) > 1 else items[0]

    def __len__(self):
        return len(self.lists[0])
    
    def push(self, *new_items):
        assert len(new_items) == len(self.lists), "New items must match the number of stored lists."
        assert len(set([len(x) for x in new_items])) == 1, "Every input must be of the same length"
        
        for i in range(len(self.lists)):
            if isinstance(self.lists[i], list):
                self.lists[i].extend(new_items[i])
                if self.max_len and len(self.lists[i]) > self.max_len:
                    self.lists[i] = self.lists[i][-self.max_len:]
            elif isinstance(self.lists[i], torch.Tensor):
                self.lists[i] = torch.cat([self.lists[i], new_items[i]], dim=0)
                if self.max_len and self.lists[i].size(0) > self.max_len:
                    self.lists[i] = self.lists[i][-self.max_len:]
            else:
                raise TypeError(f"Unsupported data type: {type(self.lists[i])}. Expected list or torch.Tensor.")

class RunningMeanStd:
    def __init__(self):
        self.mean = 0
        self.var = 1
        self.count = 0

    def update(self, x):
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)  # Ensure unbiased is set to False to match population variance
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class StatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = None

    def update(self, advantages, *args):
        # Convert input to tensor if not already
        # if not isinstance(advantages, torch.Tensor):
        #     advantages = torch.tensor(advantages)

        # Move tensor to the same device as the stats (if needed)
        # advantages = advantages.to(self.stats[0].device) if len(self.stats) > 0 else advantages
        if self.stats is None:
            self.stats = advantages
        else:
            self.stats = torch.cat((self.stats, advantages), 0)[-self.buffer_size:]

        if len(self.stats) < self.min_count:
            mean = torch.mean(advantages)
            std = torch.std(advantages) + 1e-6
        else:
            mean = torch.mean(self.stats)
            std = torch.std(self.stats) + 1e-6

        return mean, std

def explained_variance(y_pred, y_true):
    var_y = (y_true).var() + 1e-5
    return 1 - (y_true - y_pred).var() / var_y

class CosineScheduler:
    def __init__(self, base=0.996, ceil=1, total_steps=1000):
        self.tau_base = base
        self.tau_max = ceil
        self.total_steps = total_steps
        self.current_step = 0

    def current_value(self):
        current_step = min(self.current_step, self.total_steps)
        cosine_decay = (math.cos(math.pi * current_step / self.total_steps) + 1) / 2
        tau = self.tau_max - (self.tau_max - self.tau_base) * cosine_decay
        return tau

    def step(self):
        self.current_step += 1

    def reset(self):
        self.current_step = 0