from collections import deque
import random
from typing import Iterator
from torch.utils.data.dataset import IterableDataset

class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size) -> None:
        super().__init__()
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator:
        for experience in self.buffer.sample(self.sample_size):
            yield experience