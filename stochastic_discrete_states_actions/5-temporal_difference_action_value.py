#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# temporal_difference_action_value.py - script to implement a stochastic 
# discrete reinforcement learning algorithm with action value abordage.
#
# created: 2021-09-21
# author: albert r. carnier guedes (albert@teko.net.br)
#
import torch
from environment import Environment
import environment_tables as tables

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
gamma = 0.9

# Probability of exploration.
epsilon = 0.2

# Learn rate.
alpha = 0.1

# Initialize all value-states as 0.
Q = torch.zeros([env.get_n_states(), env.get_n_actions()],dtype=torch.float)

#
# TRAIN
# 
state_initial = 0
state_final   = 3

MAX_EPISODES = 1000
TOTAL_REWARD = torch.zeros(MAX_EPISODES,dtype=torch.float)

for episode in range(MAX_EPISODES):

    state = state_initial
    while state != state_final:

        # Get the max action.
        if torch.rand(1) > epsilon:
            max_action = env.rand_state_action(state)
        else:
            max_action = torch.argmax(Q[state,])

        # Get the next state.
        next_state = env.transition(state,max_action)

        # Get the reward and save to compute the total at the end of episode.
        r = env.reward(state,max_action,next_state)
        TOTAL_REWARD[episode] = TOTAL_REWARD[episode] + r

        # Get max Q of the next state.
        max_Q = torch.max(Q[next_state,])

        # Calculate the Q value of action on state.
        Q[state,max_action]+= alpha * ( r + gamma * max_Q - Q[state,max_action] )

        state = next_state

#
# FIND THE OPTIMAL POLICY.
#
policy = torch.zeros([env.get_n_states()],dtype=torch.long)
for state in env.get_states():
    policy[state] = torch.argmax(Q[state,])

#
# PRINT RESULT
#

print("\nMean Reward:",TOTAL_REWARD.mean().item())
print("Std  Reward:",TOTAL_REWARD.std().item())
print("Max  Reward:",TOTAL_REWARD.max().item())

print("\nAction-Value (Q) Table:")
print(Q)

print("\nOptimal Policy:")
state = torch.tensor([state_initial])
for state in range(tables.N_STATES):
    action = policy[state]
    print("State: ",state,", Action: ", action.item())

exit("\nDone!\n")
