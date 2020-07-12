# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:39:02 2020
PS: 程序没有跑通， 原因不不明
@author: dell
"""
import numpy as np
import gym
from my_RL_brain import PolicyGradient
import matplotlib.pyplot as plt
#%%
DISPLAY_REWARD_THRESHOLD = 400
RENDER = False


env = gym.make("CartPole-v0")
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)
print(env.observation_space.shape)
#%%

RL = PolicyGradient(
    env_name = "CartPole-v0",
    n_actions = env.action_space.n,
    n_features = env.observation_space.shape[0],
    learning_rate = 0.001,
    reward_decay = 0.99,
    output_graph = True,
    save_model = True
    )
#%%

for i_episode in range(3000):
    state = env.reset()

    while True:
        if RENDER:
            env.render()
        action = RL.choose_action(state)
        next_state, reward, done, info = env.step(action)
        RL.store_transition(state, action, reward)

        if done:
            ep_rs_sum = np.sum(RL.ep_rs)

            if "running_reward" not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = 0.99*running_reward + 0.01*ep_rs_sum

            print("episode: ", i_episode, " reward: ", running_reward)
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True

            vt = RL.learn(i_episode)
            # if i_episode % 50 == 0:
            #     plt.figure()
            #     plt.plot(vt, label="step_value")
            #     plt.xlabel("episode steps")
            #     plt.ylabel("normalized state-action value")
            #     plt.legend()
            #     plt.show()
            # pass
            break
        state = next_state
#%%
state = env.reset()

# import tensorflow.compat.v1 as tf



# tf_obs = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]],
#                          name= "observations")
# tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
# tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

# layer = tf.layers.dense(
# # self.layer = tf.Keras.layers.Dense(

#     inputs = tf_obs,
#     units = 10,
#     activation = tf.nn.tanh, # tanh activation
#     kernel_initializer = tf.random_normal_initializer(mean=0.5, stddev=0.3),
#     bias_initializer = tf.constant_initializer(0.5),
#     name = "fc1"#,
#     # reuse=tf.AUTO_REUSE
#     )

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# sess.run([layer], feed_dict={tf_obs : state[np.newaxis, :]})


#%%

# while True:
#     if RENDER: env.render()

#     action = RL.choose_action(state)

#     observation_, reward, done, info = env.step(action)

#     RL.store_transition(observation, action, reward)
#%%