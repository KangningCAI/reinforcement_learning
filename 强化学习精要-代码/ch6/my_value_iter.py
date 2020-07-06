# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:16:15 2020

@author: KangningCAI
"""
import numpy as np
from contextlib import contextmanager
import time

from my_snake import SnakeEnv, TableAgent, eval_game
from my_policy_iter import PolicyIteration


#%%
@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print("{} cost: {}".format(name, end - start) )

#%%
class PolicyIterationWithTimer(PolicyIteration):
    def policy_iteration(self, max_iter=-1):
        iteration = 0
        while True:
            iteration += 1
            
            with timer("Timer Policy Eval"):
                self._policy_evaluation(max_iter)
            
            with timer("Timer Policy Improve"):
                update = self._policy_improvement()
            
            if not update:
                break
            
        print("Iters: {} rounds, converged!".format(iteration))

#%%
class ValueIteration(TableAgent):
    def value_iteration(self, max_iter=-1):
        iter_num = 0
        while True:
            iter_num += 1
            new_value_pi = np.zeros_like(self.value_pi)
            for state in range(1,self.state_num):
                value_sas = []
                for act in range(0, self.action_num):
                    value_sa = np.dot(self.p[act, state, :], self.reward + self.gamma * self.value_pi)
                    value_sas.append(value_sa)
                    
                new_value_pi[state] = max(value_sas)
            diff = np.sqrt(np.sum(np.power(self.value_pi - new_value_pi, 2)))
            if diff < 1e-6:
                break
            else:
                self.value_pi = new_value_pi
            
            if iter_num == max_iter:
                break
        print("Iter {} rounds converged".format(iter_num))
        for state in range(1, self.state_num):
            for act in range(self.action_num):
                self.value_q[state, act] = np.dot(self.p[act, state, :], self.reward + self.gamma*self.value_pi)
            max_act = np.argmax(self.value_q[state, :])
            self.pi[state] = max_act

#%%
def value_iter_demo():
    np.random.seed(0)
    env = SnakeEnv(10, [3,6])
    agent = ValueIteration(env)
    agent.value_iteration()
    
    total_reward = eval_game(env, agent)
    print("total_reward = {}".format(total_reward))
    print("agent value_iteration.pi = {}".format(agent.pi))

#%%
def policy_iter_demo():
    np.random.seed(0)
    env = SnakeEnv(10, [3,6])
    agent = PolicyIterationWithTimer(env)
    agent.policy_iteration()
    
    total_reward = eval_game(env, agent)
    print("total_reward = {}".format(total_reward))
    print("agent policy_iteration.pi = {}".format(agent.pi))
    
#%%    
if __name__ == "__main__":
    value_iter_demo()
    policy_iter_demo()
            