#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input, Lambda, concatenate
from keras import backend as K
import time

class Agent:
    def __init__(self,
                 args,
                 sess,
                 action_interval=4,
                 no_op_steps=30,                # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
                 test_epsilon=0.05):

        self.env_name = args.env_name
        self.load = args.load
        self.save_network_path = args.network_path
        self.num_episodes = args.num_episodes
        self.num_actors = args.num_actors
        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.state_length = args.state_length
        self.n_step = args.n_step
        self.gamma = args.gamma
        self.gamma_n = self.gamma**self.n_step

        self.test_epsilon = test_epsilon

        self.action_interval = action_interval
        self.no_op_steps = no_op_steps

        self.num = 0
        self.env = gym.make(args.env_name)
        self.num_actions = self.env.action_space.n
        self.t = 0
        self.repeated_action = 0

        with tf.variable_scope("learner_parameters", reuse=True):
            self.s, self.q_values, q_network = self.build_network()

        self.q_network_weights = self.bubble_sort_parameters(q_network.trainable_weights)

        self.sess = sess

        self.sess.run(tf.global_variables_initializer())

        with tf.device("/cpu:0"):
            self.saver = tf.train.Saver(self.q_network_weights)

        # Load network
        if self.load:
            self.load_network()

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.save_network_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def bubble_sort_parameters(self, arr):
        change = True
        while change:
            change = False
            for i in range(len(arr) - 1):
                if arr[i].name > arr[i + 1].name:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    change = True
        return arr


    def build_network(self):
        l_input = Input(shape=(4,84,84))
        conv2d = Conv2D(32,8,strides=(4,4),activation='relu', data_format="channels_first")(l_input)
        conv2d = Conv2D(64,4,strides=(2,2),activation='relu', data_format="channels_first")(conv2d)
        conv2d = Conv2D(64,3,strides=(1,1),activation='relu', data_format="channels_first")(conv2d)
        fltn = Flatten()(conv2d)
        v = Dense(512, activation='relu', name="dense_v1")(fltn)
        v = Dense(1, name="dense_v2")(v)
        adv = Dense(512, activation='relu', name="dense_adv1")(fltn)
        adv = Dense(self.num_actions, name="dense_adv2")(adv)
        y = concatenate([v,adv])
        l_output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(self.num_actions,))(y)
        model = Model(input=l_input,output=l_output)

        s = tf.placeholder(tf.float32, [None, self.state_length, self.frame_width, self.frame_height])
        q_values = model(s)

        return s, q_values, model

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (self.frame_width, self.frame_height)) * 255)
        state = [processed_observation for _ in range(self.state_length)]
        return np.stack(state, axis=0)


    def preprocess(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (self.frame_width, self.frame_height)) * 255)
        return np.reshape(processed_observation, (1, self.frame_width, self.frame_height))


    def get_action_at_test(self, state):
        action = self.repeated_action

        if self.t % self.action_interval == 0:
            if random.random() <= self.test_epsilon:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        self.t += 1

        return action


