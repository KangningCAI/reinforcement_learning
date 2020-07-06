# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:12:18 2020

@author: dell
"""


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from my_snake import eval_game, SnakeEnv, ModelFreeAgent, TableAgent
from my_policy_iter import PolicyIteration
from my_monte_carlo import MonteCarlo, timer
from my_sarsa import SARSA

#%%

if __name__ == "__main__":
    """
    探索 total_reward 和 epsilon 之间的关系
    gamma : 1
    epsilon : 0~1
    """

    # for i in range(100):
    env = SnakeEnv(10, [3,6])
    dic_agent = {}
    # env_result = []

    agent_ref = PolicyIteration(env)
    agent_ref.policy_iteration()
    # total_reward_ref = eval_game(env, agent_ref)
    print(agent_ref.pi)
    dic_agent["policy_ref"] = agent_ref

    for epsilon in np.arange(0,1,.1):
        # env = SnakeEnv(10, [3,6])
        agent = SARSA(env, gamma=1, epsilon=epsilon)
        agent.sarsa()
        # total_reward = eval_game(env, agent)
        print(agent.pi)
        dic_agent["agent_"+str(epsilon)] = agent

    #%%
    # columns = ["policy_iter_ref"] +["gamma_{}".format(gamma) for gamma in np.linspace(0,1,11)]
    # df_result_epsilon = pd.DataFrame(total_env_result, columns=columns)
    df = pd.DataFrame([])
    df["agent_name"] = dic_agent.keys()
    df["agent"] = dic_agent.values()
    df["reward_list"] = df["agent"].apply(lambda agent: [eval_game(env, agent) for i in range(1000)])
    df["reward_mean"] = df["reward_list"].apply(lambda x: np.mean(x))
    df["reward_std"] = df["reward_list"].apply(lambda x: np.std(x))


    # plt.figure()
    # df.apply(lambda x: plt.hist(x["reward_list"], label=x["agent_name"]), axis=1)
    # plt.legend()
