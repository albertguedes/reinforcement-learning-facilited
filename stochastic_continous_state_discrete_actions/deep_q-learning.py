#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# deep_qlearning.py - script to implement a stochastic reinforcement 
#                     learning algorithm with discrete states and actions
#                     using deep learning and q-learnin algorithms.
#
# created: 2021-09-19
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
GAMMA = 0.9
# Learn rate.
LEARN_RATE = 10**(-4)

# Initialize all value-states as 0.
Q = torch.zeros([tables.N_STATES,tables.N_ACTIONS],dtype=torch.float)

#
# TRAIN
#

# Create the neural network.
n_inputs    = 1
n_io_hidden = 2
n_outputs   = 1
model = torch.nn.Sequential(
    torch.nn.Linear( n_inputs, n_io_hidden ),
    torch.nn.Linear( n_io_hidden, n_outputs ),
    torch.nn.ReLU()
)
# Select the loss function.
loss = torch.nn.MSELoss()
# Select the optmizer ( the method to train the model, like gradient descent or ADAM )
# SGD is the stochastic gradient descent. 
optimizer = torch.optim.ADAM( model.parameters(), lr = LEARN_RATE )

state_initial = 0
state_final = 3

MAX_EPOCHS = 100

for epoch in range(MAX_EPOCHS):

    state = state_initial
    while state != state_final:

        for action in range(tables.N_ACTIONS):

            for next_state in range(tables.N_STATES):

                # Get max Q on the next state for current action.
                max_Q = 0.0
                for next_action in range(tables.N_ACTIONS): 
                    if Q[next_state,next_action] > max_Q:
                        max_Q = Q[next_state,next_action]
    
                # Calculate the Q value of action on state.
                p = env.probability(state,action,next_state)
                r = env.reward(state,action,next_state)

                Q[state,action]+= p * ( r + GAMMA * max_Q )

        # Get a random action.
        action = torch.randint( tables.N_ACTIONS,(1,))

        # Get the next state.
        state = env.transition(state,action)

#
# FIND THE OPTIMAL POLICY.
#
policy = torch.zeros([tables.N_STATES],dtype=torch.long)
for state in range(tables.N_STATES):
    max_Q = Q[state,0]
    for action in range(tables.N_ACTIONS):
        if Q[state,action] > max_Q:
            max_Q = Q[state,action]
            policy[state] = action

#
# PRINT RESULT
#
print("\nAction-Value (Q) Table:")
print(Q)

print("\nOptimal Policy:")
state = torch.tensor([state_initial])
for state in range(tables.N_STATES):
    action = policy[state]
    print("State: ",state,", Action: ", action.item())

exit("\nDone!\n")
