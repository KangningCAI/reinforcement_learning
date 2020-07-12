# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:11:01 2020

@author: dell
"""

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf_ori

np.random.seed(1)
tf.set_random_seed(1)

#%%
class PolicyGradient(object):
    def __init__(self,env_name, n_actions, n_features, learning_rate=0.01,
                 reward_decay=0.95, output_graph=False, save_model=True):
        self.env_name = env_name
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.output_graph = output_graph
        self.save_model = save_model

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self._build_net()
        self.running_reward = tf.reduce_sum(self.tf_vt)

        self.sess = tf.Session()
        self.sess.run([ tf.global_variables_initializer(),
                        tf.local_variables_initializer() ])

        if self.output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("reward", self.running_reward)

            self.merge_summary = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter("logs_{}/".format(self.env_name), self.sess.graph)
            # tf.summary.FileWriter("logs/", self.loss)



        if self.save_model:
            self.saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state("saved_models_{}".format(self.env_name))
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded : {}".format(checkpoint.model_checkpoint_path))
            else:
                print("could not find old model weigths")





    def _build_net(self):
        with tf.name_scope("inputs"):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name= "observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # fc1
        self.layer = tf.layers.dense(
        # self.layer = tf.Keras.layers.Dense(
            inputs = self.tf_obs,
            units = 10,
            activation = tf.nn.tanh, # tanh activation
            kernel_initializer = tf.random_normal_initializer(mean=0.5, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.5),
            name = "fc1"#,
            # reuse=tf.AUTO_REUSE
            )
        # fc2
        self.all_act = tf.layers.dense(
            inputs = self.layer,
            units = self.n_actions,
            activation = None,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name = "fc2"#,
            # reuse=tf.AUTO_REUSE
            )

        self.all_act_prob = tf.nn.softmax(self.all_act, name="act_prob")

        with tf.name_scope("loss"):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only has minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act, labels=self.tf_acts) # this is negtive log of chosen action
            self.loss = tf.reduce_mean(neg_log_prob * self.tf_vt) # reward guided loss

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        all_act, prob_weights, layer, obs = self.sess.run([self.all_act, self.all_act_prob, self.layer, self.tf_obs],
                                     feed_dict={self.tf_obs : observation[np.newaxis, :]})
        # print("obs : ", obs, "dtype=", type(obs), obs.dtype)
        # print("layers : ", layer,"dtype=", type(layer), layer.dtype)

        # print("all_act : ", all_act, "dtype=", type(all_act), all_act.dtype)
        # print("prob_weights : ", prob_weights, "dtype=", type(prob_weights), prob_weights.dtype)

        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self, step, ):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # train on episode
        # self._get_running_reward()

        self.sess.run(self.train_op, feed_dict={self.tf_obs : np.vstack(self.ep_obs), # shape=[None, n_obs]
                                       self.tf_acts : np.array(self.ep_as), # shape=[None, ]
                                       self.tf_vt : discounted_ep_rs_norm, # shape=[None, ]
                                       })
        if self.output_graph:
            train_summary = self.sess.run(self.merge_summary,
                                         feed_dict={self.tf_obs : np.vstack(self.ep_obs), # shape=[None, n_obs]
                                                    self.tf_acts : np.array(self.ep_as), # shape=[None, ]
                                                    self.tf_vt : discounted_ep_rs_norm, # shape=[None, ]
                                                    })
            self.train_writer.add_summary(train_summary, step)
        if self.save_model:
            if step % 50 == 0 and step > 0:
                self.saver.save(self.sess, "saved_models_{}".format( self.env_name) + "/policy_gradients", global_step = step)
        # empty episode data
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)

        running_add = 0
        for t in reversed(range(0, len(self.ep_rs)) ):
            running_add = running_add *self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalized episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)

        return discounted_ep_rs

    # def _get_running_reward(self):
    #     ep_rs_sum = sum(self.ep_rs)
    #     if self.running_reward == 0 :
    #         self.running_reward = ep_rs_sum
    #     else:
    #         self.running_reward = self.running_reward*0.99 + ep_rs_sum*0.01






