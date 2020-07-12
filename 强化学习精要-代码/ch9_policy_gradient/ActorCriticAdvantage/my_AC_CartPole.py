# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:29:54 2020

@author: dell
"""

import numpy as np
import tensorflow as tf_ori
import tensorflow.compat.v1 as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)

#%%
# sequence parameter
SAVE_MODEL = True
OUTPUT_GRAPH = True
MAX_EPISODE = 3000
# render the env if the total episode reward > this threshold
DISPLAY_REWARD_THRESHOLD = 200
MAX_EP_STEPS = 1000 # maxium time step in one episode
RENDER = False
GAMMA = 0.9 # reward discount in TD error
LR_A = 0.001 # learning rate for actor
LR_C = 0.01

env = gym.make("CartPole-v0")
env.seed(1)
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

#%%
class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope("Actor", reuse=tf.AUTO_REUSE):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 20, # number of hidden units
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(0.0, 0.1), # weights
                bias_initializer = tf.constant_initializer(0.1), # biases
                name = "l1"
                )
            self.act_prob = tf.layers.dense(
                inputs = l1,
                units = n_actions, # output units
                activation = tf.nn.softmax, # get activation probility
                kernel_initializer = tf.random_normal_initializer(0.0, 0.1), # weights
                bias_initializer = tf.constant_initializer(0.1), # biases
                name = "acts_prob"
                )
        # print(self.act_prob.shape)
        with tf.variable_scope("exp_v", reuse=tf.AUTO_REUSE):
            log_prob = tf.log(self.act_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)
            # minimize(-self.epx_v)  = maximize(self.exp_v)


    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s : s, self.a : a, self.td_error : td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict=feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.act_prob, feed_dict={self.s : s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())




#%%
class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, "r")

        with tf.variable_scope("Critic", reuse=tf.AUTO_REUSE):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                bias_initializer = tf.constant_initializer(0.1),
                name = "l1"
                )

            self.v = tf.layers.dense(
                inputs = l1,
                units=1,
                activation = None,
                kernel_initializer = tf.random_normal_initializer(0.0, 0.1),
                bias_initializer = tf.constant_initializer(0.1),
                name = "V"
                )

        with tf.variable_scope("squared_TD_error", reuse=tf.AUTO_REUSE):
            self.td_error = self.r + GAMMA*self.v_ - self.v
            self.loss = tf.square(self.td_error)

        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s : s_})

        td_error, _ = self.sess.run([self.td_error, self.train_op], feed_dict={
            self.s : s,
            self.v_ : v_,
            self.r : r
            })
        return td_error


#%%
if __name__ == "__main__":
    # state = env.reset()
    # env.render()
    # for i in range(100):
    #     next_state, reward, done, info = env.step(1)
    #     env.render()
    #     if done:
    #         next_state = env.reset()
    #     state = next_state
    pass
    pass
#%%

if __name__ == "__main__":
    sess = tf.InteractiveSession()

    actor = Actor(sess, N_F, N_A, lr=LR_A)
    critic = Critic(sess, N_F, lr=LR_C)

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    if SAVE_MODEL:
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_models")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded : {}".format(checkpoint.model_checkpoint_path))
        else:
            print("could not find old model weigths")


    for i_episode in range(MAX_EPISODE):
        state = env.reset()
        t=0
        track_reward = []

        while True:
            if RENDER:
                env.render()
            act = actor.choose_action(state)
            next_state, reward, done, info = env.step(act)
            if done:
                reward = -20
            track_reward.append(reward)

            # gradient = grad[ reward + gamma * V(next_state) - V(state)]
            td_error = critic.learn(state, reward, next_state)
            actor.learn(state, act, td_error)


            state = next_state
            t += 1
            if done or t >= MAX_EPISODE:
                ep_rs_sum = np.sum(track_reward)

                if "running_reward" not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward*0.95 + ep_rs_sum*0.05

                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True

                print("episode: ", i_episode, "reward:", running_reward)
                if i_episode % 100 == 0 and SAVE_MODEL:
                    saver.save(sess, "saved_models/my_CartPole-v0_ActorCritic__", global_step=i_episode)
                break











