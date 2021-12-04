#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# action_value.py - script to implement a deterministic discrete reinforcement 
#                   learning algorithm with action value abordage.
#
# created: 2021-09-13
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
# Prepare to fight.
env = Environment(tables.STATES,tables.REWARDS,tables.TRANSITIONS)

# Discount factor.
gamma = 0.9

# Sett the initial and final state.
state_initial = 0
state_final   = 3

# Initialize all action-values as 0.
Q = torch.rand([tables.N_STATES,tables.N_ACTIONS],dtype=torch.float)

# Initialize final state value.
V_final = torch.rand(1,dtype=torch.float)

#
# TRAIN
#
MAX_EPISODES = 10

TOTAL_REWARD = torch.zeros(MAX_EPISODES,dtype=torch.float)

for episode in range(MAX_EPISODES):

    # Calculate the value state foreach state.
    for state in env.get_states():

        # Get actions with permitted TRANSITIONS from the state.
        current_actions = env.actions(state)

        for action in current_actions:

            # Get the next state from the current action.
            next_state = env.transition(state,action)

            max_Q = 0.0
            if next_state != state_final:

                # Get actions of the next state with permitted TRANSITIONS.
                current_next_actions = env.actions(next_state)
                for next_action in current_next_actions:
                    if Q[next_state,next_action] > max_Q:
                        max_Q = Q[next_state,next_action]

            else:
                
                # V(S') = max{Q(S',a')}
                max_Q = V_final

            # Get the reward.
            r = env.reward(state,action,next_state)
            
            # Q(S,a) = r(S,a,S') + gamma * max{Q(S',a')}
            Q[state,action] = r + gamma * max_Q

        # Get the reward.
        r = env.reward(state,action,next_state)
        TOTAL_REWARD[episode]+= r

        

#
# FIND THE OPTIMAL POLICY.
#

# Define policy tensor. All actions are prohibitive (-1) by default.
policy = -torch.ones(tables.N_STATES,dtype=torch.long)

state = state_initial
while state != state_final :

    # Get actions with permitted TRANSITIONS from the state.
    current_actions = env.actions(state)

    max_Q = 0.0
    for action in current_actions:
        if Q[state,action] > max_Q:
            max_Q = Q[state,action]
            policy[state] = action

    state = env.transition(state,policy[state])

#
# PRINT RESULTS
#
print("\nMean Reward:",TOTAL_REWARD.mean().item())
print("Std  Reward:",TOTAL_REWARD.std().item())
print("Max  Reward:",TOTAL_REWARD.max().item())

print("\nAction-Value Table:")
print(Q)

print("\nOptimal Policy:")
state = torch.tensor([state_initial])
while state != state_final:
    action = policy[state]
    print("State: ",state.item(),", Action: ", action.item())
    state = env.transition(state,action)

exit("\nDone!\n")