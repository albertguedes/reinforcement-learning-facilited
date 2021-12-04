#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# created: 2021-09-21
# author: albert r. carnier guedes (albert@teko.net.br)
#
import torch
from memory import Memory

MEMORY_SIZE = 10
EXPERIENCE_DIMENSION = 3
BATCH_SIZE  = 3

# Set he reply memory.
memory = Memory(MEMORY_SIZE,EXPERIENCE_DIMENSION)

print("Dump:",memory.dump())
print("memory size:", memory.size())

# Add experiences to memory.
for i in range(MEMORY_SIZE):
    experience = torch.rand(EXPERIENCE_DIMENSION)
    print(i,". ",experience)
    memory.push(experience)
    print("memory size:", memory.size())

print("Dump:",memory.dump())
print("memory size:", memory.size())

samples = memory.samples(BATCH_SIZE)

print("\nSamples:\n",samples)

memory.clear()

print("Dump:",memory.dump())
print("memory size:", memory.size())


exit(0)