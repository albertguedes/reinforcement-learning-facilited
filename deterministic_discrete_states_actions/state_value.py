#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# state_value.py - script to implement a deterministic discrete reinforcement 
#                  learning algorithm with state value abordage.
#
# created: 2021-09-07
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
env = Environment(tables.STATES,tables.REWARDS,tables.TRANSITIONS)

# Discount factor.
gamma = 0.9

# Initialize all value-states as 0.
V = torch.rand(tables.N_STATES,dtype=torch.float)

#
# TRAIN
# 

MAX_EPISODES = 100

TOTAL_REWARD = torch.zeros(MAX_EPISODES,dtype=torch.float)

for episode in range(MAX_EPISODES):

    # Calculate the value state foreach state.
    for state in env.get_states():

        # Get actions with permitted TRANSITIONS from the state.
        current_actions = env.actions(state)

        # Find the value state for each action on the state.
        max_V = 0.0
        for action in current_actions:

            # Get the next state.
            next_state = env.transition(state,action)

            # Get and save the reward.
            r = env.reward(state,action,next_state)
            TOTAL_REWARD[episode] = TOTAL_REWARD[episode] + r

            # V(S) = r(S,A,S') + gamma * V(S')
            max_V = r + gamma * V[next_state]

            # If the max_value is great then current state value of the state
            # then, update the state value
            if max_V > V[state]:
                V[state] = max_V

#
# FIND THE OPTIMAL POLICY.
#

# Define policy tensor. They store the optimal actions on a given state.
# All actions are prohibitive (-1) by default.
policy = -torch.ones(tables.N_STATES,dtype=torch.long)

# Calculate the value state foreach state.
state_initial = 0
state_final   = 3

# Follow the states with the biggest state-values to find the optimal policy.
state = state_initial
while state != state_final :

    # Get actions on the state.
    current_actions = env.actions(state)

    # Select the action that arrive on a new state with biggest state-value.
    max_V = 0.0
    for action in current_actions:
        next_state = env.transition(state,action)
        if V[next_state] > max_V:
            max_V = V[next_state]
            policy[state] = action

    state = env.transition(state,policy[state])

#
# PRINT RESULTS
#
print("\nMean Reward:",TOTAL_REWARD.mean().item())
print("Std  Reward:",TOTAL_REWARD.std().item())
print("Max  Reward:",TOTAL_REWARD.max().item())

print("\nState-Value Table:")
for state in range(len(V)):
    print("V(",state,") =",V[state].item())

print("\nOptimal Policy:")
state = torch.tensor([state_initial])
while state != state_final:
    action = policy[state]
    print("State: ",state.item(),", Action: ", action.item())
    state = env.transition(state,action)

exit("\nDone!\n")
