# environment.py - simulate a environment with discrete states, that receives a discrete
#                  action and an state and return a new state and a reward.
#
# created: 2021-09-16
# author: albert r. carnier guedes (albert@teko.net.br)
#

import os
import sys
#sys.path.append('/home/albert/Projetos/PYTHON/p.machinelearning/models/reinforcement_learning/stochastic_discrete_states_actions/Environments')
sys.path.append('./Environments')

import torch
import importlib

#
# Class to interact with the environment.
#
class Environment:

    def __init__( self, environment_name ):
        # Set environmnt dfined by environment name.
        env = importlib.import_module(environment_name)
        self._states_table      = env.STATES
        self._n_states          = torch.tensor(env.STATES.size(dim=0),dtype=torch.long)
        self._actions_table     = env.ACTIONS
        self._n_actions         = torch.tensor(env.ACTIONS.size(dim=0),dtype=torch.long)
        self._rewards_table     = env.REWARDS
        self._transitions_table = env.TRANSITIONS

    # Get the list of states.
    def states(self):
        return self._states_table

    # Get the total number of states.
    def n_states(self):
        return self._n_states

    # Get the list of actions.
    def actions(self):
        return self._actions_table

    # Get the total number of actions.
    def n_actions(self):
        return self._n_actions

    # Get the transitions table.
    def transitions(self):
        return self._transitions_table

    # Get the rewards table.
    def rewards(self):
        return self._rewards_table

    # Get the next state given the actual state and the action executed on 
    # this state.
    def next_state(self, state = None, action = None ):
        if ( state != None ) and ( action != None ):
            x = torch.rand(1)
            p_low = 0.0
            for next_state in self._states_table:
                p = self._transitions_table[state,action,next_state]
                if p > 0.0:
                    p_high = p_low + p
                    if p_low <= x < p_high :
                        return next_state
                    p_low = p_high
            return state
        else:
            print("ERROR on Environment.get_next_state : state or action equal to 'None'")
            return False            

    # Get the reward of a given transition.
    def reward(self, state = None, action = None, next_state = None ):
        if ( state != None ) and ( action != None ) and ( next_state != None ):
            return self._rewards_table[state,action,next_state]
        else:
            print("ERROR on Environment.reward : state, action or next_state is equal to 'None'")
            return False            

    # Get the actions permitted on given state.
    def state_actions(self, state = None ):
        if state != None:
            possibles = []
            for action in self.actions():
                p = torch.sum( self._transitions_table[state,action,:] )
                if p > 0.0:
                    possibles.append(action)
            return torch.tensor(possibles)
        else:
            print("ERROR on Environment.state_actions: state is equal to 'None'")
            return False

    # Get a random possible action of a given state.
    def rand_action(self,state = None):
        if state != None:
            possibles = self.state_actions(state)
            idx = torch.randint(len(possibles),(1,) )
            return possibles[idx]
        else:
            print("ERROR on Environment.rand_action: state is equal to 'None'")
            return False

    # Get the probability of a given transition.
    def probability( self, state, action, next_state ):
        if ( state != None ) and ( action != None ) and ( next_state != None ):
            return self._transitions_table[ state, action, next_state ]
        else:
            print("ERROR on Environment.probability: state, action or next_state is equal to 'None'")
            return False            
