# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:04:46 2020

@author: dell
"""

import gym
from my_RL_brain import PolicyGradient
import matplotlib.pyplot as plt

# render the env if total episode reward is greater than this  threshold
DISPLAY_REWARD_THRESHOLD = -2000

# episode: 537   reward: -2216
# episode: 538   reward: -2224
# episode: 539   reward: -2216
# ...
# episode: 997   reward: -373
# episode: 998   reward: -370
# episode: 999   reward: -370


# rendering waste time
RENDER = False
# RENDER = True


env = gym.make("MountainCar-v0")
env.seed(1) # reproducible, general Policy Gradient has high variance
env = env.unwrapped
# 据说不做这个动作会有很多限制，unwrapped是打开限制的意思
# *** attention : env = env.unwrapped, 如果没有重新赋值，程序的reward始终固定为-200 (PS)   ***

print(env.action_space)
# 查看这个环境中可用的action有多少个，返回Discrete()格式
n_actions=env.action_space.n
# 查看这个环境中可用的action有多少个，返回int

print(env.observation_space)# 查看这个环境中observation的特征，返回Box()格式
print(env.observation_space.high)
print(env.observation_space.low)
n_features=env.observation_space.shape[0]
# 查看这个环境中observation的特征有多少个，返回int
#%%
RL = PolicyGradient(env_name="MountainCar-v0", n_actions=n_actions, n_features=n_features,
                    learning_rate=0.02, reward_decay=0.995, output_graph=False, save_model=False)

#%%
# %matplotlib inline
for i_episode in range(1000):
# i_episode = 0
# while True:
    # i_episode += 1
    observation = env.reset()

    while True:
        if RENDER:
            env.render()
        action = RL.choose_action(observation)
        # reward is -1 in all cases
        observation_, reward, done, info = env.step(action)
        RL.store_transition(observation, action, reward)

        if done:
            # calcuate running reward
            ep_rs_sum = sum(RL.ep_rs)
            if "running_reward" not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward*0.99 + ep_rs_sum*0.01

            # if running_reward > DISPLAY_REWARD_THRESHOLD:
            #     RENDER = True

            print("episode : {}, reward : {}".format(int(i_episode), int(running_reward)))

            # train()
            vt = RL.learn(i_episode)
            if i_episode % 30 == 0:
                plt.figure()
                plt.plot(vt)
                plt.xlabel("episode_steps")
                plt.ylabel("normalized state-action value")
                plt.show()

            break
        observation = observation_

