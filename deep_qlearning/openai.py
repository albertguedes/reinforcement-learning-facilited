#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# openai.py - script to implement open ai gym environments.
#
# created: 2021-10-03
# author: albert r. carnier guedes (albert@teko.net.br)
#
import gym
import time

env = gym.make('CartPole-v0')

print(env.observation_space)
print()
print( (env.observation_space.shape)[0])

episodes = range(200)

for episode in episodes:

    state = env.reset()

    done  = False

    step = 0
    while not done :

        env.render()
        action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)

        print(state, action, next_state, reward, done, info)

        if done:
            print("\nEpisode finished after {} timesteps.\n".format(step+1))
            break
    
        state = next_state
        step+= 1

    time.sleep(1)

env.close()