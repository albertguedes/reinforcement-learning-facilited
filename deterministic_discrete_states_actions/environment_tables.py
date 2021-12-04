# set_environment.py - prepare the environment for a given simulation.
#
# created: 2021-09-16
# author: albert r. carnier guedes (albert@teko.net.br)
#

import torch 

# Set the number of states.
N_STATES = 4

# Set the states.
STATES = torch.tensor(range(N_STATES),dtype=torch.long)

# Set the number of actions.
N_ACTIONS = 2

# Set the actions.
ACTIONS = torch.tensor(range(N_ACTIONS),dtype=torch.long)

# Transition from a state S to S', executing a action A on S.
# S' = transition(S,A)
# Transition with -1 value are forbiden.
TRANSITIONS = -torch.ones([N_STATES,N_ACTIONS],dtype=torch.long)
# Set the TRANSITIONS.
TRANSITIONS[0,0] = 1
TRANSITIONS[0,1] = 2
TRANSITIONS[1,0] = 3
TRANSITIONS[2,0] = 1
TRANSITIONS[2,1] = 3

# Reward for starting from state S, performing action a, and arriving in state s':
# R = REWARDS(S,A,S')
# Set all the REWARDS as -1 by default.
REWARDS = -torch.ones([N_STATES,N_ACTIONS,N_STATES],dtype=torch.long)
# Set the REWARDS.
REWARDS[0,0,1] = 2
REWARDS[0,1,2] = 1
REWARDS[1,0,3] = 3
REWARDS[2,0,1] = 4
REWARDS[2,1,3] = 5
