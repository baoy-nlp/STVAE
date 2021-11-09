"""
    Generating sentence from the continuous space of *VAE
"""

import torch.nn as nn


class SequenceGenerator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        pass
