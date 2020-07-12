# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:27:16 2020

@author: dell
"""

import tensorflow.compat.v1 as tf
import numpy as np
import gym

np.random.seed(2)
tf.set_random_seed(2) # reproducible


#%%
class Actor(object):
    def __init__(self, sess, n_features, action_bound, lr=0.001):
        self.sess = sess

        self.state = tf.placeholder(tf.float32, [1, n_features], "state")
        self.action = tf.placeholder(tf.float32, None, "action")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")

        l1 = tf.layers.dense(
            inputs = self.state,
            units = 30,
            activation = tf.nn.relu,
            kernel_initializer = tf.random_normal_initializer(0.0, 0.1), # weights
            bias_initializer = tf.constant_initializer(0.1), # biases
            name = "l1"
            )

        mu = tf.layers.dense(
            inputs = l1,
            units = 1,
            activation = tf.nn.tanh,
            kernel_initializer = tf.random_normal_initializer(0.0, 0.1),
            bias_initializer = tf.constant_initializer(0.1),
            name = "mu"
            )

        sigma = tf.layers.dense(
            inputs = l1,
            units = 1,
            activation = tf.nn.softplus,
            kernel_initializer = tf.random_normal_initializer(0.0, 0.1),
            bias_initializer = tf.constant_initializer(1.0),
            name = "sigma"
            )

        global_step = tf.Variable(0, trainable=False)

        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+0.1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])

        with tf.name_scope("exp_v"):
            log_prob = self.normal_dist.log_prob(self.action) # loss without advantage
            self.exp_v = log_prob * self.td_error # advantage (TD error) guided loss
            self.exp_v += 0.01 * self.normal_dist.entropy()

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)

    def learn(self, state, action, td):
        state = state[np.newaxis, :]
        feed_dict = {self.state : state,
                     self.action : action,
                     self.td_error : td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, state):
        state = state[np.newaxis, :]
        return self.sess.run(self.action, {self.state: state})

#%%
class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        with tf.name_scope("inputs"):
            self.state = tf.placeholder(tf.float32, [1, n_features], name="state")
            self.value_next = tf.placeholder(tf.float32, [1,1], name="value_next")
            self.reward = tf.placeholder(tf.float32, name="reward")

        with tf.variable_scope("Critic"):
            l1 = tf.layers.dense(
                inputs = self.state,
                units = 30,
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(0.0, 0.1),
                bias_initializer = tf.constant_initializer(0.1),
                name = "l1"
                )

            self.value = tf.layers.dense(
                inputs = l1,
                units = 1,
                activation = None,
                kernel_initializer = tf.random_normal_initializer(0.0, 0.1),
                bias_initializer = tf.constant_initializer(0.1),
                name = "value"
                )

        with tf.variable_scope("squared_TD_error"):
            self.td_error = tf.reduce_mean(self.reward + GAMMA*self.value_next - self.value)
            self.loss = tf.square(self.td_error)


        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, state, reward, state_next):
        state, state_next = state[np.newaxis, :], state_next[np.newaxis, :]
        value_next = self.sess.run(self.value, feed_dict={self.state : state_next})

        td_error, _ = self.sess.run([self.td_error, self.train_op], feed_dict={
            self.state : state,
            self.reward : reward,
            self.value_next : value_next,
            })

        return td_error

#%%
OUTPUT_GRAPH = True
SAVE_MODEL = True

MAX_EPISODE = 1000*5
MAX_EP_STEPS = 200
DISPLAY_REWARD_THRESHOLD = -100
RENDER = False

GAMMA = 0.9
LR_A = 0.001
LR_C = 0.01

#%%
env = gym.make("Pendulum-v0")
env.seed(1)
env = env.unwrapped

N_S = env.observation_space.shape[0]
A_BOUND = env.action_space.high

sess = tf.InteractiveSession()

actor = Actor(sess, n_features=N_S, action_bound=[-A_BOUND, A_BOUND], lr=LR_A)
critic = Critic(sess, n_features=N_S, lr=LR_C)

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    writer = tf.summary.FileWriter("log_AC_continue_Pendulum/", sess.graph)

if SAVE_MODEL:
    saver = tf.train.Saver()
    save_path = "saved_models__AC_continue_Pendulum/"
    checkpoint = tf.train.get_checkpoint_state(save_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded : {}".format(checkpoint.model_checkpoint_path))
    else:
        print("could not find old model weigths")
#%%
for i_episode in range(MAX_EPISODE):
    state = env.reset()
    t = 0
    ep_rs = []

    while True:
        if RENDER:
            env.render()
        action = actor.choose_action(state)
        state_next, reward, done, info = env.step(action)

        # gradient = grad[r + gamma * V(s_) - V(s)]
        td_error = critic.learn(state=state, reward=reward, state_next=state_next)

        # true_gradient = grad[logPi(s,a)*td_error]
        actor.learn(state=state, action=action, td=td_error)


        state = state_next
        t += 1
        ep_rs.append(reward)

        if t > MAX_EP_STEPS:
            ep_rs_sum = sum(ep_rs)
            if "running_reward" not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward*0.9 + ep_rs_sum*0.1

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True

            print("episode:{}, reward:{}".format(i_episode, running_reward))
            if SAVE_MODEL and i_episode % 300 == 0 :
                saver.save(sess, save_path+"/ActorCritic_", global_step = i_episode)
            break











