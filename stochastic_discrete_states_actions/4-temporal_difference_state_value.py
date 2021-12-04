#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# temporal_difference_sate_value.py - script to implement a stochastic 
# discrete reinforcement learning algorithm with state value abordage.
#
# created: 2021-09-21
# author: albert r. carnier guedes (albert@teko.net.br)
#
import torch
from environment import Environment

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
env = Environment('environment0')

actions = env.actions()
states  = env.states()

state_initial = 0
state_final   = 3

#
# TRAIN
# 

# Discount factor.
GAMMA = 1.0

# Probability of exploration.
EPSILON = 0.4

# Learn rate.
ALPHA = 0.01

# Initialize all value-states as 0.
V = torch.zeros([env.n_states()],dtype=torch.float)

MAX_EPISODES = 1000
MAX_STEPS    = 10
TOTAL_REWARD = torch.zeros(MAX_EPISODES,dtype=torch.float)
TOTAL_STEPS  = torch.zeros(MAX_EPISODES,dtype=torch.float)

episodes = range(MAX_EPISODES)

for episode in episodes:

    state = state_initial
    step = 0
    while ( state != state_final ) and ( step < MAX_STEPS ):

        # Get the max action.
        if torch.rand(1) < EPSILON:

            # Get max V of the next state.
            max_V = 0.0
            for action_ in env.state_actions(state):
                next_state_ = env.next_state(state,action_)
                if V[next_state_] > max_V:
                    max_V = V[next_state_]

            # Get a random action.
            action = env.rand_action(state)
            # Get the next state.
            next_state = env.next_state(state,action)

        else:

            # Find the action that maximize V.
            max_V = 0.0
            for action_ in env.state_actions(state):
                next_state_ = env.next_state(state,action_)
                if V[next_state_] > max_V:
                    max_V      = V[next_state_]
                    action     = action_
                    next_state = next_state_

        # Get the reward and save to compute the total at the end of episode.
        r = env.reward(state,action,next_state)
        TOTAL_REWARD[episode] = r + GAMMA * TOTAL_REWARD[episode] 

        print(V)
        
        # Calculate the Q value of action on state.
        V[state]+= ALPHA * ( r + GAMMA * max_V - V[state] )

        state = next_state
        step+= 1

    TOTAL_STEPS[episode] = step

    print("Episode",episode,".:. Steps",TOTAL_STEPS[episode].item(),".:. Reward",TOTAL_REWARD[episode].item())

#
# PRINT RESULT
#
print("\nState-Value Table:")
for state in env.states():
    print("V(",state.item(),") =",V[state].item())

print("\nSteps per episode:")
print("Mean Steps:",TOTAL_STEPS.float().mean().item())
print("Std  Steps:",TOTAL_STEPS.float().std().item())
print("Max  Steps:",TOTAL_STEPS.float().max().item())

print("\nRewards per episode:")
print("Mean Reward:",TOTAL_REWARD.mean().item())
print("Std  Reward:",TOTAL_REWARD.std().item())
print("Max  Reward:",TOTAL_REWARD.max().item())

#
# FIND THE OPTIMAL POLICY.
#
policy = torch.zeros([env.n_states()],dtype=torch.long)

for state in env.states():
    # Find the action that maximize V.
    max_V = 0.0
    for action_ in env.state_actions(state):
        next_state_ = env.next_state(state,action_)
        if V[next_state_] > max_V:
            max_V         = V[next_state_]
            policy[state] = action_

print("\nOptimal Policy:")
policy_reward = 0.0
state = state_initial
step = 0
while ( state != state_final ) and ( step < MAX_STEPS ):
    action     = policy[state]
    next_state = env.next_state(state,action)
    r          = env.reward(state,action,next_state)
    policy_reward = r + GAMMA * policy_reward
    print("State:",state.item(),".:. Action:", action.item(),".:. Next State:", next_state.item(), ".:. Step reward:",policy_reward.item())
    state = next_state
    step+=1

print("\nOptimal Policy Reward: ",policy_reward.item())

exit("\nDone!\n")
