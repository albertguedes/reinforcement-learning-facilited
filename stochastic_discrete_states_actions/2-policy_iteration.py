#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# policy_iterattion.py - script to implement a stochastic discrete reinforcement 
#                        learning algorithm with policy iteration abordage.
#
# created: 2021-09-17
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

actions = env.get_actions()
states  = env.get_states()

#
# TRAIN
# 

# Discount factor.
GAMMA = 0.9

# Initialize all value-states as 0.
V = torch.zeros(env.get_n_states(),dtype=torch.float)

# Maximum error of state-values.
ERROR = 1.0e-3
# DELTA ensure that the greatest error on value-state is less than ERROR.
DELTA = 1.0

MAX_EPISODES = 10000
MAX_STEPS    = 100
TOTAL_REWARD = torch.tensor(MAX_EPISODES, dtype=torch.float)
TOTAL_STEPS  = torch.tensor(MAX_EPISODES, dtype=torch.float)

episode = 0
while ( DELTA >= ERROR ) and ( episode < MAX_EPISODES ) :

    DELTA = 0.0

    V_current = torch.tensor(0.0,dtype=torch.float)

    # Calculate de value-state foreach state.
    step = 0
    for state in states:

        # Save current state value.
        V_current = V[state]

        # Find the max value-state between the actions.
        for action_ in actions:

            mean_V = 0.0
            for next_state_ in states:
                p = env.probability(state,action_,next_state_)
                if p > 0.0:
                    r = env.reward(state,action_,next_state_)
                    mean_V+= p * ( r + GAMMA * V[next_state_] )

            if mean_V > V[state]:
                V[state]   = mean_V
                action     = action_

        next_state = env.next_state(state,action)
        
        r = env.reward(state,action,next_state)
        TOTAL_REWARD[episode] = r + TOTAL_REWARD[episode]

        # Difference between old V and max V: |v-V(S)|
        diff = abs(torch.sub(V_current,V[state]))

        DELTA = max(DELTA,diff)

        step+= 1

        if step > MAX_STEPS:
            break

    episode+= 1

#
# FIND THE OPTIMAL POLICY.
#

# Initialize the policy tensor with random actions.
policy = torch.zeros(env.get_n_states,dtype=torch.long)

for state in states:

    # Find the max value-state between the actions.
    max_V = torch.tensor(0.0,dtype=torch.long)
    for action in actions:

        sum_V = torch.tensor(0.0,dtype=torch.long)
        for next_state in states:
            p = env.probability(state,action,next_state)
            r = env.reward(state,action,next_state)
            sum_V+= p * ( r + GAMMA * V[next_state] )

        if sum_V > max_V:
            max_V = sum_V
            policy[state] = action

#
# PRINT RESULT
#
print("\nState-Value Table:")
for state in states:
    print("V(",state,") =",V[state].item())

print("\nOptimal Policy:")
for state in states:
    action = policy[state]
    print("State: ",state,", Action: ", action.item())

exit("\nDone!\n")
