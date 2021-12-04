#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# action_value.py - script to implement a stochastic discrete reinforcement 
#                   learning algorithm with SARSA abordage.
#
# created: 2021-09-17
# author: albert r. carnier guedes (albert@teko.net.br)
#
import torch
from environment import Environment
import environment_tables as tables

from matplotlib import pyplot
import seaborn

#####################################################################
#                              CONVENTIONS                          #
#                                                                   #
# 1. Implicitly, the states are labeled as 0,1,2,3,..., N_STATES    #
#    and the actions are labeled as 0,1,...,N_ACTIONS.              #
#                                                                   #
# 2. For infinite horizon, set final_state = -1                     #
#                                                                   #
#####################################################################

# Prepare to fight.
env = Environment(tables.STATES,tables.ACTIONS,tables.REWARDS,tables.TRANSITIONS)

# Discount factor.
GAMMA   = torch.tensor(1.0,dtype=torch.float)
# Probability of risc.
EPSILON = torch.tensor(0.0,dtype=torch.float)
# Learning rate.
ALPHA   = torch.tensor(1e-5,dtype=torch.long)

# Initialize all value-states as 0.
Q = torch.rand([env.get_n_states(), env.get_n_actions()],dtype=torch.float)

#
# TRAIN
# 
state_initial = torch.tensor(0,dtype=torch.long)
state_final   = torch.tensor(3,dtype=torch.long)

MAX_EPISODES  = torch.tensor(10000,dtype=torch.long)
MAX_STEPS     = torch.tensor(10,dtype=torch.long)
TOTAL_REWARDS = torch.zeros([MAX_EPISODES],dtype=torch.float)

for episode in range(MAX_EPISODES):

    step  = 0
    state = state_initial
    while state != state_final and step < MAX_STEPS :

        # Get current action.
        action = 0
        if torch.rand(1) < EPSILON:
            action = torch.randint( env.get_n_actions(), (1,))
        else:
            action = torch.argmax(Q[state,])

        # Get next state.
        next_state = env.transiction(state, action)

        # Get current reward.
        r = env.reward(state,action,next_state)
        TOTAL_REWARDS[episode] = r + GAMMA * TOTAL_REWARDS[episode]

        # Get next action.
        next_action = 0
        if torch.rand(1) < EPSILON:
            next_action = torch.randint( env.get_n_actions(), (1,))
        else:
            next_action = torch.argmax(Q[state,])

        Q[state,action] = Q[state,action] + ALPHA * ( r + GAMMA * Q[next_state,next_action] - Q[state,action] )

        # Get the next state.
        state = next_state

        step+= 1

#
# PRINT TOTAL REWARD.
#
print("\nMean Reward:",TOTAL_REWARDS.mean().item() )
print("Std  Reward:",  TOTAL_REWARDS.std().item()  )
print("Max  Reward:",  TOTAL_REWARDS.max().item()  )

#
# PRINT RESULT
#
print("\nAction-Value (Q) Table:")
print(Q)

print("\nOptimal Policy:")
optimal_reward = torch.tensor(0.0,dtype=torch.float)

step  = 0
state = state_initial
while state != state_final and step < MAX_STEPS :

    action = torch.argmax(Q[state,])

    # Get next state.
    next_state = env.transiction(state, action)

    # Get current reward.
    r = env.reward(state,action,next_state)
    optimal_reward = r + GAMMA * optimal_reward

    print("State:",state.item(),"Action:",action.item(),"Next State:",next_state.item())

    # Get the next state.
    state = next_state

    step+= 1

print("\nOptimal Reward: ",optimal_reward.item())

#
# PLOT THE GRAPH
#
x = range(MAX_EPISODES)
y = TOTAL_REWARDS

pyplot.hist(y)
pyplot.show()

exit("\nDone!\n")
