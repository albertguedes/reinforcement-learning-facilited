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
import gym
import time
from nn import NN

#####################################################################
#                              CONVENTIONS                          #
#                                                                   #
# 1. Implicitly, the states are labeled as 0,1,2,3,..., N_STATES    #
#    and the actions are labeled as 0,1,...,N_ACTIONS.              #
#                                                                   #
# 2. For infinite horizon, set final_state = -1                     #
#                                                                   #
#####################################################################

###############
# Environment #
###############

#   Type: Box(4)
#        Num     Observation               Min                     Max
#        0       Cart Position             -4.8                    4.8
#        1       Cart Velocity             -Inf                    Inf
#        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
#        3       Pole Angular Velocity     -Inf                    Inf

# Actions:
#        Type: Discrete(2)
#        Num   Action
#        0     Push cart to the left
#        1     Push cart to the right

# Prepare to fight.
env = gym.make('CartPole-v0')

################
# Reply Memory #
################

##################
# Neural Network #
##################

n_inputs    = state_dimension
n_io_hidden = 256
n_outputs   = n_actions

# Set gradient descent learning rate.
LEARNING_RATE = 1e-3

# The PREDICT neural network.
Q_pred = NN(n_inputs,n_outputs,n_io_hidden)

# Set the optimizer of weights of the neural network.
optimizer = torch.optim.Adam( Q_pred.parameters(), lr = LEARNING_RATE )

# The TARGET neural network.
Q_target = NN(n_inputs,n_outputs,n_io_hidden)

# Clone PREDICT nn to TARGET nn.
Q_target.load_state_dict( Q_pred.state_dict() )

# Select the loss function.
loss = torch.nn.SmoothL1Loss()

##############
# Q-Learning #
############## 

# Discount factor.
GAMMA = 1e0

# Max episodes of iteration with environment. 
MAX_EPISODES = 1000
# Max steps to the agent. 
MAX_STEPS = 10*MAX_EPISODES

# Save total rewards received on one episode. 
rewards_per_episode = torch.zeros(MAX_EPISODES,dtype=torch.float)

# Get the range of episodes and steps.
episodes = range(MAX_EPISODES)

for episode in episodes:

    # Get initial state.
    state = torch.tensor( env.reset() ,dtype=torch.float)

    # Not done yet.
    done = False

    # Begin the sequence of steps.
    step = 0
    while ( not done ) and ( step < MAX_STEPS ):

        env.render()

        # Get predicted action.
        action = torch.argmax( Q_pred(state) ).item()
        
        # Get next state, reward and verify final state.
        next_state, reward, done, info = env.step(action)

        rewards_per_episode[episode] = reward + GAMMA * rewards_per_episode[episode]

        if done:
            print("Episode finished after {:4d} timesteps".format(step+1))
            break

        step+= 1
        
        state = torch.tensor( next_state ,dtype=torch.float)

    steps_per_episode[episode] = step 

    time.sleep(1)

env.close()

#
# PRINT RESULT
#
print("\nMean Steps:",steps_per_episode.mean().item())
print(  "Std  Steps:",steps_per_episode.std().item())
print(  "Max  Steps:",steps_per_episode.max().item())

print("\nMean Reward:",rewards_per_episode.mean().item())
print(  "Std  Reward:",rewards_per_episode.std().item())
print(  "Max  Reward:",rewards_per_episode.max().item())

exit("\nDone!\n")
