# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 15:14:53 2020

@author: KangningCAI
"""

import numpy as np
from my_snake import SnakeEnv, TableAgent, eval_game
from my_value_iter import PolicyIterationWithTimer, ValueIteration, timer

#%%
class GeneralizedPolicyIteration(PolicyIterationWithTimer, ValueIteration):
    # def __init__(self, env):
    #     self.env = env
    #     self.table = TableAgent(env)
    #     self.agent_policy_iter = PolicyIterationWithTimer(self.env)
    #     self.agent_value_iter = ValueIteration(self.env)
        
    def generalized_policy_iteration(self):
        # self.agent_policy_iter.policy_iteration()
        # self.agent_value_iter.value_iteration()
        self.policy_iteration()
        print(self.pi)
        self.value_iteration()
        print(self.pi)
#%%
def policy_iteration_demo():
    np.random.seed(0)
    env = SnakeEnv(10, [3,6])
    agent = PolicyIterationWithTimer(env)
    with timer("Timer PolicyIter "):
        agent.policy_iteration()
    print("return_pi = {}".format(eval_game(env, agent)))
    
def value_iteration_demo():
    np.random.seed(0)
    env = SnakeEnv(10, [3,6])
    agent = ValueIteration(env)
    with timer("Timer ValueIter"):
        agent.value_iteration()
    print("return_pi = {}".format(eval_game(env, agent)))

def generalized_iteration_demo():
    np.random.seed(0)
    env = SnakeEnv(10, [3,6])
    agent = GeneralizedPolicyIteration(env)
    with timer("Timer GeneralizedIter"):
        agent.generalized_policy_iteration()
    print("return_pi = {}".format(eval_game(env, agent)))



#%%
if __name__ == "__main__":
    policy_iteration_demo()
    value_iteration_demo()
    generalized_iteration_demo()
    