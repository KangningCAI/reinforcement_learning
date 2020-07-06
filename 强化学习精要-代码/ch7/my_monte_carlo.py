# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:13:13 2020

@author: KangningCAI
"""
import numpy as np
from contextlib import contextmanager
import time

from my_snake import SnakeEnv, TableAgent, ModelFreeAgent, eval_game
import gym
from my_policy_iter import PolicyIteration


#%%
@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print("{} cost: {}".format(name, end - start) )

#%%
class MonteCarlo(object):
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon
    
    def monte_carlo_eval(self, agent, env):
        state = env.reset()
        episode = []
        while True:
            act = agent.play(state, self.epsilon)
            next_state, reward, terminate, info = env.step(act)
            episode.append((state, act, reward))
            
            state = next_state
            if terminate:
                break
        
        value = []
        return_value = 0
        for item in reversed(episode):
            return_val = return_value * agent.gamma + item[2]
            value.append((item[0], item[1], return_val))
            
        for state, act, accu_value in reversed(value):
            agent.value_n[state][act] += 1
            agent.value_q[state][act] += (accu_value - agent.value_q[state][act]) / agent.value_n[state][act]
            
    def policy_improve(self, agent):
        """
        

        Parameters
        ----------
        agent : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            if successfully update agent.pi, return True.
            else return False, which means the updating process has ended.

        """
        new_policy = np.zeros_like(agent.pi)
        for state in range(1, agent.state_num):
            new_policy[state] = np.argmax(agent.value_q[state, :])
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True
        
    def monte_carlo_opt(self, agent, env):
        for i in range(10):
            for j in range(100):
                self.monte_carlo_eval(agent, env)
            self.policy_improve(agent)

#%%
def monte_carlo_demo():
    np.random.seed(101)
    env = SnakeEnv(10, [3,6])
    agent = ModelFreeAgent(env)
    mc = MonteCarlo()
    with timer("Timer Monte Carlo Iter"):
        mc.monte_carlo_opt(agent, env)
    print("reward = {}".format(eval_game(env,agent)))
    print("return pi = {}".format(agent.pi))
    
def policy_iter_demo():
    np.random.seed(101)
    env = SnakeEnv(10, [3,6])
    agent = PolicyIteration(env)
    with timer("Timer PolicyIter"):
        agent.policy_iteration()
    print("reward = {}".format(eval_game(env,agent)))
    print("return pi = {}".format(agent.pi))
    
def monte_carlo_demo2():
    np.random.seed(101)
    env = SnakeEnv(10, [3,6])
    agent = ModelFreeAgent(env)
    mc = MonteCarlo(.5)
    with timer("Timer monte_carlo_demo2"):
        mc.monte_carlo_opt(agent, env)
    print("reward = {}".format(eval_game(env,agent)))
    print("return pi = {}".format(agent.pi))
    
    
#%%
if __name__ == "__main__":
    monte_carlo_demo()  
    policy_iter_demo()
    monte_carlo_demo2()    
    #%%
    # np.random.seed(101)
    env = SnakeEnv(10, [3,6])
    agent = ModelFreeAgent(env)
    mc = MonteCarlo(1)
    with timer("Timer monte_carlo_demo2"):
        mc.monte_carlo_opt(agent, env)
    print("reward = {}".format(eval_game(env,agent)))
    print("return pi = {}".format(agent.pi))     
        
