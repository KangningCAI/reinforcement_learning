# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 14:17:19 2020

@author: dell
"""

import numpy as np
import gym
from gym.spaces import Discrete
#%%
class SnakeEnv(gym.Env):
    SIZE = 100
    def __init__(self, ladder_num, dices):
        self.ladder_num = ladder_num
        self.dices = dices
        self.ladders = dict(np.random.randint(1, self.SIZE, size=(self.ladder_num,2) ) )
        self.observation_space = Discrete(self.SIZE+1)
        self.action_space = Discrete( len(dices) )

        for k,v in list(self.ladders.items()):
            self.ladders[v] = k

        self.pos = 1

    def reset(self):
        self.pos = 1
        return self.pos

    def reward(self, s):
        if s == 100:
            return 100
        else:
            return -1


    def step(self, a):
        step = np.random.randint(1, self.dices[a]+1 )

        self.pos += step
        if self.pos == 100:
            return self.pos, self.reward(self.pos), 1, {}
        elif self.pos > 100:
            self.pos = 200 - self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]

        return self.pos, self.reward(self.pos), 0, {}

    def render(self):
        pass

#%%
class TableAgent(object):
    def __init__(self, env):
        self.state_num = env.observation_space.n
        self.action_num = env.action_space.n

        self.reward = [env.reward(state) for state in range(0, self.state_num)]
        self.pi = np.array([0 for state in range(0, self.state_num) ])
        self.p = np.zeros([self.action_num, self.state_num, self.state_num])

        ladder_move = np.vectorize(lambda x:  env.ladders[x] if x in env.ladders else x)

        for i,dice in enumerate(env.dices):
            prob = 1.0/dice

            for src in range(1,100):
                step = np.arange(dice)
                step += src
                step = np.piecewise(step, [step>100, step<=100],
                                    [lambda x: 200-x, lambda x: x]
                                    )
                step = ladder_move(step)
                for dst in step:
                    self.p[i, src, dst] += prob

        self.p[:, 100, 100] = 1
        self.value_pi = np.zeros((self.state_num))
        self.value_q = np.zeros((self.state_num, self.action_num))
        self.gamma = 0.8

    def play(self, state):
        return self.pi[state]

#%%
class ModelFreeAgent(object):
    def __init__(self, env):
        self.state_num = env.observation_space.n
        self.action_num = env.action_space.n
        
        self.pi = np.array([0 for state in range(0, self.state_num)])
        self.value_q = np.zeros((self.state_num, self.action_num))
        self.value_n = np.zeros((self.state_num, self.action_num))
        self.gamma = 0.8

    def play(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_num)
        else:
            # return np.argmax( self.value_q[state, :] )
            return self.pi[state]
#%%
def eval_game(env, policy):
    state = env.reset()
    accumulate_reward = 0

    while True:
        if isinstance(policy, list):
            act = policy[state]
        elif isinstance(policy, TableAgent):
            act = policy.play(state)
        elif isinstance(policy, ModelFreeAgent):
            act = policy.play(state)
        else:
            raise Exception("Illegal policy")

        state, reward, terminate, info = env.step(act)
        accumulate_reward += reward
        if terminate:
            break
    return accumulate_reward

#%%
if __name__ == "__main__":
    print("1. test SnakeEnv()")
    env = SnakeEnv(10, [3,6])
    env.reset()
    while True:
        state, reward, terminate, info = env.step(0)
        print(state, reward )
        if terminate == 1:
            break
    #%%
    print("2. test eval_game() with simple policy")
    env = SnakeEnv(10, [3,6])
    policy_ref = [1]*97 + [0]*3
    policy_0 = [0]*100
    policy_1 = [1]*100

    sum_ref = 0
    sum_0 = 0
    sum_1 = 0
    for i in range(10000):
        sum_ref += eval_game(env, policy_ref)
        sum_0 += eval_game(env, policy_0)
        sum_1 += eval_game(env, policy_1)
    print("result_ref: avg={}".format(sum_ref/10000))
    print("result_0: avg={}".format(sum_0/10000))
    print("result_1: avg={}".format(sum_1/10000))


    #%%
    print("3. test  TableAgent")
    print("-"*50)
    env = SnakeEnv(0, [3,6])
    agent = TableAgent(env)

    total_reward = eval_game(env, agent)
    print(agent.pi)
    print("init : total_reward = {}".format(total_reward))
    
    agent.pi[:] = 0
    total_reward = eval_game(env, agent)
    print("all 0 : total_reward = {}".format(total_reward))
    
    agent.pi[:] = 1
    total_reward = eval_game(env, agent)
    print("all 1 : total_reward = {}".format(total_reward))
    
    agent.pi[97:100] = 0
    total_reward = eval_game(env, agent)
    print("ref: total_reward = {}".format(total_reward))
    
    



