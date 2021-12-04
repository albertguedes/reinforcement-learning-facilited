#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# action_value.py - script to implement a stochastic discrete reinforcement 
#                   learning algorithm with state value abordage.
#
# created: 2021-09-17
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
GAMMA = 0.3

# Probability of exploration.
EPSILON = 0.1

# Initialize all action-values randomly.
Q = torch.zeros([env.n_states(), env.n_actions()],dtype=torch.float)

MAX_EPISODES = 1000
MAX_STEPS    = 100
TOTAL_REWARD = torch.zeros(MAX_EPISODES,dtype=torch.float)
TOTAL_STEPS  = torch.zeros(MAX_EPISODES,dtype=torch.long)

episodes = range(MAX_EPISODES)

for episode in episodes:

    state = state_initial
    step = 0

    while ( state != state_final ) and ( step < MAX_STEPS ):

        # Selec method tto retrieve action. 
        # Random action for exploration
        # Max Arg for exploitation.
        if torch.rand(1) < EPSILON:
            # Get a random possible action on state.
            action = env.rand_action(state)                    
        else:
            # For each possible action on state, compute the Q-value.
            for action in env.state_actions(state):
                for next_state in states:
                    # Get the probabiliy transition.
                    # If p = 0.0, arent transition and arent computation of Q.
                    p = env.probability(state,action,next_state)
                    if p > 0.0 :
                        # Get the reward transiction.
                        r = env.reward(state,action,next_state)
                        # Calculate the Q value of action on state following the 
                        # the Bellman equation
                        # Q(s,a) = max_a < r(s,a,s') + GAMMA * max_a' { Q(s',a') } >
                        Q[state,action]+= p * ( r + GAMMA * torch.max(Q[next_state,]) )

            # Get the action of the max action-value.
            action = torch.argmax(Q[state,])

        # Get the next state.
        next_state = env.next_state(state,action)

        # Get the reward.
        r = env.reward(state,action,next_state)
        TOTAL_REWARD[episode] = r + GAMMA * TOTAL_REWARD[episode] 

        state = next_state
        step+= 1

    TOTAL_STEPS[episode] = step

    print("Episode",episode,".:. Steps",TOTAL_STEPS[episode].item(),".:. Reward",TOTAL_REWARD[episode].item())

#
# PRINT RESULTS.
#
print("\nAction-Value (Q) Table:")
print(Q)

print("\nSteps per episode:")
print("Mean Steps:",TOTAL_STEPS.float().mean().item())
print("Std  Steps:",TOTAL_STEPS.float().std().item())
print("Max  Steps:",TOTAL_STEPS.float().max().item())

print("\nRewards per episode:")
print("\nMean Reward:",TOTAL_REWARD.mean().item())
print("Std  Reward:",TOTAL_REWARD.std().item())
print("Max  Reward:",TOTAL_REWARD.max().item())

#
# FIND THE OPTIMAL POLICY.
#
policy_reward = 0.0
policy = torch.zeros([env.n_states()],dtype=torch.long)
print("\nOptimal Policy:")

state = state_initial
step = 0
while ( state != state_final ) and ( step < MAX_STEPS ):

    policy[state] = torch.argmax(Q[state,])

    next_state = env.next_state(state,policy[state])

    r = env.reward(state,policy[state],next_state)
    
    policy_reward = r + GAMMA * policy_reward

    print("State:",state.item(),".:. Action:", policy[state].item(),".:. Next State:",next_state.item(), ".:. Step reward:",policy_reward.item())

    state = next_state
    step+=1

print("State:",state.item(),".:. Action:", policy[state].item(),".:. Next State:",next_state.item(), ".:. Step reward:",policy_reward.item())

print("\nOptimal Policy Reward:",policy_reward.item())

exit("\nDone!\n")
