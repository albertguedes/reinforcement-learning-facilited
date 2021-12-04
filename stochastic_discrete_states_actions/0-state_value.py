#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# state_value.py - script to implement a stochastic discrete reinforcement 
#                  learning algorithm with state value abordage.
#
# created: 2021-09-16
# author: albert r. carnier guedes (albert@teko.net.br)
#
import torch
import time
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

# List of states.
states  = env.states()
# List of actions.
actions = env.actions()

# Set initial and final state.
state_initial = states[0]
state_final   = states[-1]

#
# TRAIN
# 

# Discount factor.
GAMMA = 0.9

MAX_EPISODES = 1000
MAX_STEPS    = 10
episodes     = range(MAX_EPISODES)

# Initialize value-states with random values.
V = torch.rand(env.n_states(),dtype=torch.float)

# Save total rewards of each episode.
TOTAL_REWARD = torch.zeros(MAX_EPISODES,dtype=torch.float)
# Save total steps of each episode.
TOTAL_STEPS = torch.zeros(MAX_EPISODES,dtype=torch.long)

for episode in episodes:

    state = state_initial
    step  = 0

    while ( state != state_final ) and  ( step < MAX_STEPS ) :

        # Find max V following the Bellman Equation.
        # V(S) = max_a { < r(S,a,S') + gamma * V(S') > }
        max_V = 0.0

        # Iterate on each possible action on state.
        for action in env.state_actions(state) :

            # Compute the mean value of each next state.
            mean_V = 0.0
            for next_state in states:
                p = env.probability(state,action,next_state)
                if p > 0.0:
                    r = env.reward(state,action,next_state)
                    mean_V+= p * ( r + GAMMA * V[next_state] )

            if mean_V > max_V:
                max_V = mean_V

        V[state] = max_V

        # Get a random action.
        action = env.rand_action(state)

        # Get the next state.
        next_state = env.next_state(state,action)

        r = env.reward(state,action,next_state)
        TOTAL_REWARD[episode] = r + GAMMA * TOTAL_REWARD[episode]

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

    # Find max V.
    max_V = 0.0
    for action in env.state_actions(state):

        mean_V = 0.0
        for next_state in env.states():
            p = env.probability(state,action,next_state)
            r = env.reward(state,action,next_state)
            mean_V+= p * ( r + GAMMA * V[next_state] )

        if mean_V > max_V:
            max_V = mean_V
            policy[state] = action

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
