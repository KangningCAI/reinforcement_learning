# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:10:35 2020

@author: dell
"""
import numpy as np
import pandas as pd

from my_snake import eval_game, SnakeEnv, ModelFreeAgent, TableAgent
from my_policy_iter import PolicyIteration
from my_monte_carlo import MonteCarlo, timer

#%%
class SARSA(ModelFreeAgent):
    def __init__(self, env, gamma=1, epsilon=0.0):
        super(SARSA,self).__init__(env)
        self.epsilon = epsilon
        self.env = env
        self.gamma = gamma

    def sarsa_eval(self ):
        state = self.env.reset()
        prev_state = -1
        prev_act = -1

        while True:
            act = self.play(state, self.epsilon)
            next_state, reward, terminate, info = self.env.step(act)
            next_act = self.play(next_state)
            if prev_state != -1:
                return_val = reward + self.gamma*(0 if terminate else self.value_q[next_state][next_act])
                self.value_n[state][act] += 1
                self.value_q[state][act] += (return_val - self.value_q[state][act]) / self.value_n[state][act]
                pass

            prev_state = state
            prev_act = act
            state = next_state

            if terminate:
                break

    def sarsa_improve(self):
        new_policy = np.zeros_like(self.pi)
        for state in range(1, self.state_num):
            new_policy[state] = np.argmax(self.value_q[state, :])

        if np.all(np.equal(new_policy, self.pi)):
            print("# update process has finished")
            return False
        else:
            self.pi = new_policy
            print("# update process +1 ")
            return True

    def sarsa(self):
        for i in range(10):
            for j in range(2000):
                self.sarsa_eval()
            self.sarsa_improve()




#%%
if __name__ == "__main__":

    env = SnakeEnv(10, [3,6])
    agent_ref = PolicyIteration(env)
    agent_ref.policy_iteration()
    total_reward_ref = eval_game(env, agent_ref)
    print(agent_ref.pi)

    # env = SnakeEnv(10, [3,6])
    agent = SARSA(env, gamma=1, epsilon=0)
    agent.sarsa()
    total_reward = eval_game(env, agent)
    print(agent.pi)

    # env = SnakeEnv(10, [3,6])
    agent_2 = SARSA(env, gamma=.8, epsilon=0)
    agent_2.sarsa()
    total_reward_2 = eval_game(env, agent_2)
    print(agent_2.pi)

#%%
if __name__ == "__main__":
    """
    探索 total_reward 和 gamma 之间的关系
    gamma : 0~1
    epsilon : 0
    """
    env = SnakeEnv(10, [3,6])

    agent_ref = PolicyIteration(env)
    agent_ref.policy_iteration()

    def get_sarsa(gamma, epsilon):
        print("-"*10, gamma, "-"*10)
        agent = SARSA(env, gamma=gamma, epsilon=0)
        agent.sarsa()
        return agent

    ls_agent = [agent_ref] + [get_sarsa(gamma, epsilon=0) for gamma in np.arange(0, 1, 0.1)]
    ls_agent_name = ["policy_iter_ref"] +["sarsa_gamma_{}".format(str(gamma)) for gamma in np.arange(0., 1.,0.1)]

    df_result = pd.DataFrame([])
    df_result["agent_name"] = ls_agent_name
    df_result["agent"] = ls_agent
    # df_result.loc[1:, "agent"].apply(lambda x: x.sarsa())
    df_result["reward_list"] = df_result.apply(lambda x:  [eval_game(env, x["agent"]) for i in range(1000)],
                                               axis=1)
    df_result["reward_mean"] = df_result["reward_list"].apply(lambda x: np.mean(x))
    df_result["reward_std"] = df_result["reward_list"].apply(lambda x: np.std(x))

#%%
if __name__ == "__main__":
    """
    探索 total_reward 和 epsilon 之间的关系
    gamma : 1
    epsilon : 0~1
    """
    ls_agent = [agent_ref] + [get_sarsa(gamma=1, epsilon=epsilon) for epsilon in np.arange(0, 1, 0.1)]
    ls_agent_name = ["policy_iter_ref"] +["sarsa_epsilon_{}".format(str(epsilon)) for epsilon in np.arange(0., 1.,0.1)]




    df_result_epsilon = pd.DataFrame([])
    df_result_epsilon["agent_name"] = ls_agent_name
    df_result_epsilon["agent"] = ls_agent
    df_result_epsilon["reward_list"] = df_result_epsilon.apply(lambda x:  [eval_game(env, x["agent"]) for i in range(1000)],
                                               axis=1)
    df_result_epsilon["reward_mean"] = df_result_epsilon["reward_list"].apply(lambda x: np.mean(x))
    df_result_epsilon["reward_std"] = df_result_epsilon["reward_list"].apply(lambda x: np.std(x))