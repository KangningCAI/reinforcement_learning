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

    def _policy_evaluation(self, max_iter=-1):
        iteration = 0
        # 多次迭代计算 self.value_pi
        while True:
            # one iteration
            iteration += 1
            new_value_pi = self.value_pi.copy()
            # 计算 value_pi
            for state in range(1, self.state_num): # for each state
                act = self.play(state)
                transition_p = self.p[act, state, :]
                value_sa = np.dot( transition_p, self.reward + self.gamma * self.value_pi)

                new_value_pi[state] = value_sa

            # 更新及退出条件
            diff = np.sqrt( np.sum( np.power( self.value_pi - new_value_pi, 2)))
            if diff < 1e-6:
                break
            else:
                self.value_pi = new_value_pi

            if iteration == max_iter:
                break

    def _policy_improvement(self):
        """


        Returns
        -------
        bool
            if successfuly update the policy, return Ture.
            else return False, which reprensents the policy has converged.

        """
        new_policy = np.zeros_like(self.pi)
        for state in range(1, self.state_num):
            for act in range(self.action_num):
                self.value_q[state, act] = np.dot( self.p[act, state, :], self.reward + self.gamma * self.value_pi)
            max_act = np.argmax(self.value_q[state, :])
            new_policy[state] = max_act

        if np.all(new_policy == self.pi):
            return False
        else:
            self.pi = new_policy
            return True

    def policy_iteration(self):
        iter_num = 0
        while True:
            iter_num += 1
            self._policy_evaluation()
            update = self._policy_improvement()

            if not update:
                break
        print("Iter:{} rounds, converged!".format(iter_num))


#%%
def eval_game(env, policy):
    state = env.reset()
    accumulate_reward = 0

    while True:
        if isinstance(policy, list):
            act = policy[state]
        elif isinstance(policy, TableAgent):
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
    "1. test SnakeEnv()"
    env = SnakeEnv(10, [3,6])
    env.reset()
    while True:
        state, reward, terminate, info = env.step(0)
        print(state, reward )
        if terminate == 1:
            break
    #%%
    "2. test eval_game() with simple policy"
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
    "3. test  TableAgent"
    env = SnakeEnv(0, [3,6])
    agent = TableAgent(env)

    total_reward = eval_game(env, agent)
    print(agent.pi)
    print("init : total_reward = {}".format(total_reward))


    agent.policy_iteration()
    total_reward = eval_game(env, agent)
    print(agent.pi)
    print("after policy iter : total_reward = {}".format(total_reward))

    env = SnakeEnv(10, [3,6])
    total_reward = eval_game(env, agent)
    print(agent.pi)
    print("init : total_reward = {}".format(total_reward))


    agent.policy_iteration()
    total_reward = eval_game(env, agent)
    print(agent.pi)
    print("after policy iter : total_reward = {}".format(total_reward))


