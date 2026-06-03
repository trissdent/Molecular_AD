# base.py
import torch
from abc import ABC, abstractmethod


class ModelManager(ABC):
    """
    Base class for all models.
    Inherit this with nn.Module.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def summary(self):
        print(self)
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"Total parameters: {total:,}")
        print(f"Trainable: {trainable:,}")
        print(f"Frozen: {frozen:,}")