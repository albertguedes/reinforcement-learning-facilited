#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# test_environment.py - test environment class.
#
# created: 2021-09-16
# author: albert r. carnier guedes (albert@teko.net.br)
#

import os
import sys
sys.path.append('/home/albert/Projetos/PYTHON/p.machinelearning/models/reinforcement_learning/stochastic_discrete_states_actions')

from environment import Environment 

env = Environment('environment0')

print(env.states())
print(env.actions())
print(env.n_states())
print(env.n_actions())
print(env.transitions())
print(env.rewards())

print(env.next_state(0,0))
print(env.reward(0,0,0))
print(env.probability(0,0,0))
print(env.state_actions(0))
print(env.rand_action(0))
