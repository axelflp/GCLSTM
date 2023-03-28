import sys
import os.path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Torch functions
import torch
import torch.nn as nn

from variables import HORAS_POR_PREDECIR


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()

        self.input_size = input_size

        self.perceptron = nn.Sequential(
            nn.Linear(self.input_size, 8),
            nn.Sigmoid(),
            nn.Linear(8, 48),
            nn.Sigmoid(),
            nn.Linear(48, HORAS_POR_PREDECIR)
        )

    def forward(self, X):
        return self.perceptron(X)

