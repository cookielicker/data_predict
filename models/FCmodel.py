"""
    basic fully connection model
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
           is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class mlp(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    self.norm = RMSNorm(hidden_size)
    self.up = nn.Linear(hidden_size, hidden_size * 2)
    self.act = nn.ReLU()
    self.down = nn.Linear(hidden_size * 2, hidden_size)

  def forward(self, states):
    residual = states
    states = self.norm(states)
    states = self.up(states)
    states = self.act(states)
    states = self.down(states)
    states = states + residual
    return states

class FCmodel(nn.Module):
  def __init__(self, input_size, num_class, hidden_size=256, num_layers=10):
    super().__init__()
    self.head_0 = nn.Linear(input_size, hidden_size)
    self.head_1 = nn.Linear(input_size, hidden_size)
    self.layers = nn.ModuleList([mlp(hidden_size=hidden_size) for idx in range(num_layers)])
    self.classifier = nn.Linear(hidden_size, num_class, bias=False)

  def forward(self, input_seq):
    x1, x2 = input_seq.chunk(2, dim=-1)
    x1 = self.head_0(x1)
    x2 = self.head_1(x2)
    x = x1 + x2
    for layer in self.layers:
      x = layer(x)
    x = self.classifier(x)
    return x