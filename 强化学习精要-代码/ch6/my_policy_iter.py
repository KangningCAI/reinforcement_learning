# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 08:36:49 2020

@author: KangningCAI
"""

from my_snake import SnakeEnv, TableAgent, eval_game
import numpy as np


class PolicyIteration(TableAgent):
    
    
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
if __name__ == "__main__":
    print("4. test Policy iter")
    print("-"*50)
    env = SnakeEnv(0, [3,6])
    agent = PolicyIteration(env)
    
    total_reward = eval_game(env, agent)
    print(agent.pi)
    print("init : total_reward = {}".format(total_reward))
    
    print("-"*50)
    agent.policy_iteration()
    total_reward = eval_game(env, agent)
    print(agent.pi)
    print("after policy iter : total_reward = {}".format(total_reward))

    print("\n\n" + "-"*50)
    env2 = SnakeEnv(10, [3,6])
    agent2 = PolicyIteration(env2)
    total_reward = eval_game(env2, agent2)
    print(agent2.pi)
    print("init : total_reward = {}".format(total_reward))

    print("-"*50)
    agent2.policy_iteration()
    total_reward = eval_game(env2, agent2)
    print(agent2.pi)
    print("after policy iter : total_reward = {}".format(total_reward))