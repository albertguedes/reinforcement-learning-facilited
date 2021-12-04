#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# memory.py - script with reply memory class.
#
# created: 2021-09-21
# author: albert r. carnier guedes (albert@teko.net.br)
#
import torch

######################
# Reply Memory Class #
######################
class Memory:

    def __init__(self, memory_size = 0, experience_dimension = 0 ):
            
        self._DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

        self._MEMORY_SIZE          = memory_size
        self._EXPERIENCE_DIMENSION = experience_dimension

        # Create tthe memory.
        self._memory = torch.zeros([memory_size,experience_dimension],dtype=torch.float, device=self._DEVICE)

        # Size counter.
        self._size = 0

    # Save state ( experience ) on reply memory.
    def push(self, experience ):

        # Verify if memory is full.
        if self._size < self._MEMORY_SIZE:
            # Save on the first empty row on memory.
            self._memory[self._size] = experience
            # Increment the size after add experience.
            self._size+= 1
        else:
            # If the memory is full, delete first row and shift
            # all rows to allocate the last row to add the experience.
            for i in range(self._MEMORY_SIZE-1):
                self._memory[i] = self._memory[i+1]
            self._memory[-1] = experience

    # Get random sample of experiences stored on memory.
    def samples(self, batch_size = 0 ):
        # Get a batch size numbers of experiences randomly.
        idxs = torch.randint(self._size,(batch_size,),device=self._DEVICE)
        return self._memory[idxs]

    # Return the current size of the memory.
    def size(self):
        return self._size

    # Dump the values on memory.
    def dump(self):
        return self._memory

    # Clear the memory.
    def clear(self):
        self._memory = torch.zeros([self._MEMORY_SIZE,self._EXPERIENCE_DIMENSION],dtype=torch.float,device=self._DEVICE)
        # Size counter.
        self._size = 0
