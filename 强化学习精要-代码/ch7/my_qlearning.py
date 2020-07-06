# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:15:52 2020

@author: dell
"""
import numpy as np
import pandas as pd

from my_snake import SnakeEnv, ModelFreeAgent, TableAgent, eval_game
import gym
from my_policy_iter import PolicyIteration
from my_monte_carlo import MonteCarlo, timer


#%%
class QLearning(ModelFreeAgent):
    def __init__(self, env, gamma=1, epsilon=0):
        super(QLearning, self).__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env

    def qlearning_eval(self):
        state = self.env.reset()

        while True:
            act = self.play(state)
            next_state, reward, terminate, info = self.env.step(act)
            return_val = reward + self.gamma * (0 if terminate else np.max(self.value_q[next_state, :]))
            self.value_n[state][act] +=1
            self.value_q[state][act] += (return_val - self.value_q[state][act]) / self.value_n[state][act]

            state = next_state
            if terminate:
                break

    def qlearning_improve(self):
        new_policy = np.zeros_like(self.pi)
        for state in range(1, self.state_num):
            new_policy[state] = np.argmax(self.value_q[state, :])

        if np.all(np.equal(new_policy, self.pi)):
            print("update has finished!")
            return False
        else:
            self.pi = new_policy
            print("update +1")
            return True

    def qlearning(self):
        for i in range(10):
            for j in range(3000):
                self.qlearning_eval()
            self.qlearning_improve()
#%%
if __name__ == "__main__":

    env = SnakeEnv(10, [3,6])
    #%%
    agent_ref = PolicyIteration(env)
    agent = QLearning(env, gamma=1, epsilon=0)
    agent2 = QLearning(env, gamma=1, epsilon=0.5)
    agent3 = QLearning(env, gamma=1, epsilon=1)


    agent_ref.policy_iteration()
    agent.qlearning()
    agent2.qlearning()
    agent3.qlearning()


    #%%
    total_reward_ref = eval_game(env, agent_ref)
    print(agent_ref.pi)

    total_reward = eval_game(env, agent)
    print(agent.pi)

    total_reward2 = eval_game(env, agent2)
    print(agent2.pi)

    #%%
    df = pd.DataFrame(["agent_policy_ref", "agent_qlearning_epsilon_0", "agent_qlearning_epsilon_0.5",
                       "agent_qlearning_epsilon_1" ],
                      columns=["agent_name"])
    df["agent"] = [agent_ref, agent, agent2, agent3]
    df["reward_list"] = df.apply(lambda x: [eval_game(env, x["agent"]) for i in range(1000)], axis=1)
    df["reward_mean"] = df["reward_list"].apply(lambda x: np.mean(x))
    df["reward_std"] = df["reward_list"].apply(lambda x: np.std(x))
