#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# nn.py - script with neural network class.
#
# created: 2021-09-21
# author: albert r. carnier guedes (albert@teko.net.br)
#
import torch
from torch import nn, optim

# Neural Network to aproximate the Q table.
class NN( nn.Module ):

    def __init__(self, n_inputs = 0, n_outputs = 0, n_io_hidden = 0 ):
        nn.Module.__init__(self)

        self._model = nn.Sequential(
            nn.Linear(n_inputs, n_io_hidden),
            nn.ReLU(),
            nn.Linear(n_io_hidden, n_outputs)
        )

        # Initialize the weights with random values.
        self._model.apply(self._init_weights)

    # Set weights with random values.
    def _init_weights(self,m):
        if type(m) == torch.nn.Linear:
            m.weight.data.uniform_(0.0,1.0)

    def forward(self, state):
        return self._model(state)
