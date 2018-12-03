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

class Actor:
    def __init__(self,
                 args,
                 queues,
                 number,
                 sess,
                 param_copy_interval=400,
                 send_size=50,                  # Send () transition to shared queue in a time.
                 action_interval=4,
                 no_op_steps=30,                # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode

                 epsilon=0.4,
                 alpha=7,
                 anealing=False,
                 no_anealing_steps=20000,
                 anealing_steps=1000000,
                 initial_epsilon=1.0,
                 final_epsilon=0.1):

        self.queue = queues[0]
        self.param_queue = queues[1]


        self.env_name = args.env_name
        self.num_episodes = args.num_episodes
        self.num_actors = args.num_actors
        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.state_length = args.state_length
        self.n_step = args.n_step
        self.gamma = args.gamma
        self.gamma_n = self.gamma**self.n_step

        self.param_copy_interval = param_copy_interval
        self.send_size = send_size
        self.action_interval = action_interval
        self.no_op_steps = no_op_steps

        self.epsilon = epsilon
        self.alpha = alpha

        self.anealing = anealing
        self.no_anealing_steps = no_anealing_steps
        self.anealing_steps = anealing_steps


        self.env = gym.make(args.env_name)


        self.num = number
        self.num_actions = self.env.action_space.n

        self.t = 0
        self.repeated_action = 0

        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        if not self.anealing:
            self.epsilon = self.epsilon **(1+(self.num/(self.num_actors-1))*self.alpha) if self.num_actors != 1 else self.epsilon
        else:
            self.epsilon = initial_epsilon
            self.epsilon_step = (initial_epsilon - final_epsilon)/ anealing_steps



        self.local_memory = deque(maxlen=self.send_size*2)
        self.buffer = []
        self.R = 0

        # with tf.device("/cpu:0"):
        self.s, self.q_values, q_network = self.build_network()

        self.q_network_weights = self.bubble_sort_parameters(q_network.trainable_weights)

        self.st, self.target_q_values, target_network = self.build_network()
        self.target_network_weights = self.bubble_sort_parameters(target_network.trainable_weights)

        self.a, self.y, self.q, self.error = self.td_error_op()

        learner_params = self.param_queue.get()
        shapes = self.get_params_shape(learner_params)

        self.ph_list = [tf.placeholder(tf.float32, shape=shapes[i]) for i in range(len(shapes))]
        self.target_ph_list = [tf.placeholder(tf.float32, shape=shapes[i]) for i in range(len(shapes))]
        self.obtain_q_parameters = [self.q_network_weights[i].assign(self.ph_list[i]) for i in range(len(self.q_network_weights))]
        self.obtain_target_parameters = [self.target_network_weights[i].assign(self.target_ph_list[i]) for i in range(len(self.target_network_weights))]

        self.sess = sess

        self.sess.run(tf.global_variables_initializer())

        self.sess.run([self.obtain_q_parameters,self.obtain_target_parameters],
                      feed_dict=self.create_feed_dict(learner_params))

    def create_feed_dict(self, learner_params):
        feed_dict = {}
        for i in range(len(learner_params[0])):
            feed_dict[self.ph_list[i]] = learner_params[0][i]
            feed_dict[self.target_ph_list[i]] = learner_params[1][i]
        return feed_dict


    def get_params_shape(self, learner_params):
        shapes = []
        for p in learner_params[0]:
            shapes.append(p.shape)
        return shapes

    def bubble_sort_parameters(self, arr):
        change = True
        while change:
            change = False
            for i in range(len(arr) - 1):
                if arr[i].name > arr[i + 1].name:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    change = True
        return arr


    def td_error_op(self):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])
        q = tf.placeholder(tf.float32, [None,None])
        #w = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector. shape=(BATCH_SIZE, num_actions)
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        # shape = (BATCH_SIZE,)
        q_value = tf.reduce_sum(tf.multiply(q, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)

        return a, y, q, error


    def build_network(self):
        l_input = Input(shape=(4,84,84))
        conv2d = Conv2D(32,8,strides=(4,4),activation='relu', data_format="channels_first")(l_input)
        conv2d = Conv2D(64,4,strides=(2,2),activation='relu', data_format="channels_first")(conv2d)
        conv2d = Conv2D(64,3,strides=(1,1),activation='relu', data_format="channels_first")(conv2d)
        fltn = Flatten()(conv2d)
        v = Dense(512, activation='relu', name="dense_v1_"+str(self.num))(fltn)
        v = Dense(1, name="dense_v2_"+str(self.num))(v)
        adv = Dense(512, activation='relu', name="dense_adv1_"+str(self.num))(fltn)
        adv = Dense(self.num_actions, name="dense_adv2_"+str(self.num))(adv)
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



    def get_action_and_q(self, state):
        action = self.repeated_action
        q = self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}, session=self.sess)
        if self.t % self.action_interval == 0:
            if self.epsilon >= random.random():
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(q[0])
            self.repeated_action = action

        return action, q[0]

    def get_action_at_test(self, state):
        action = self.repeated_action

        if self.t % self.action_interval == 0:
            if random.random() <= 0.05:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        self.t += 1

        return action

    def get_sample(self, n):
        s, a, _, _, _, q = self.buffer[0]
        _, _, _, s_, done, q_ = self.buffer[n-1]
        return s, a, self.R, s_, done, q, q_


    def run(self):# , server):
        for _ in range(self.num_episodes):
            terminal = False
            observation = self.env.reset()
            for _ in range(random.randint(1, self.no_op_steps)):
                last_observation = observation
                observation, _, _, _ = self.env.step(0)  # Do nothing
            state = self.get_initial_state(observation, last_observation)
            start = time.time()

            while not terminal:
                last_observation = observation
                action, q = self.get_action_and_q(state)

                observation, reward_ori, terminal, _ = self.env.step(action)
                reward = np.sign(reward_ori)
                self.env.render(mode='rgb_array')
                processed_observation = self.preprocess(observation, last_observation)
                next_state = np.append(state[1:, :, :], processed_observation, axis=0)

                self.buffer.append((state, action, reward, next_state, terminal, q))
                self.R = round((self.R + reward * self.gamma_n) / self.gamma,3)

                # n-step transition
                if terminal:      # terminal state
                    while len(self.buffer) > 0:
                        n = len(self.buffer)
                        s, a, r, s_, done, q, q_ =  self.get_sample(n)
                        self.local_memory.append((s, a, r, s_, done, q, q_))
                        self.R = round((self.R - self.buffer[0][2]) / self.gamma,3)
                        self.buffer.pop(0)
                    self.R = 0

                if len(self.buffer) >= self.n_step:
                    s, a, r, s_, done, q, q_ = self.get_sample(self.n_step)
                    self.local_memory.append((s, a, r, s_, done, q, q_))
                    self.R = self.R - self.buffer[0][2]
                    self.buffer.pop(0)

                # Add experience and priority to remote memory
                if len(self.local_memory) > self.send_size:
                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    next_state_batch = []
                    terminal_batch = []
                    q_batch = []
                    qn_batch = []

                    for _ in range(self.send_size):
                        data = self.local_memory.popleft()
                        state_batch.append(data[0])
                        action_batch.append(data[1])
                        reward_batch.append(data[2])
                        #shape = (BATCH_SIZE, 4, 32, 32)
                        next_state_batch.append(data[3])
                        terminal_batch.append(data[4])
                        q_batch.append(data[5])
                        qn_batch.append(data[6])

                    #remote_memory.extend(send)
                    terminal_batch = np.array(terminal_batch) + 0
                    # shape = (BATCH_SIZE, num_actions)
                    target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)}, session=self.sess)
                    # DDQN
                    actions = np.argmax(qn_batch, axis=1)
                    target_q_values_batch = np.array([target_q_values_batch[i][action] for i, action in enumerate(actions)])
                    # shape = (BATCH_SIZE,)
                    y_batch = reward_batch + (1 - terminal_batch) * self.gamma_n * target_q_values_batch

                    error_batch = self.error.eval(feed_dict={
                        self.s: np.float32(np.array(state_batch) / 255.0),
                        self.a: action_batch,
                        self.q: q_batch,
                        self.y: y_batch
                    }, session=self.sess)

                    send = [(state_batch[i],action_batch[i],reward_batch[i],next_state_batch[i],terminal_batch[i]) for i in range(self.send_size)]
                    self.queue.put((send,error_batch))

                state = next_state

                self.t += 1

                if self.t % self.param_copy_interval == 0:
                    while self.param_queue.empty():
                        print('Actor {} is wainting for learner params coming'.format(self.num))
                        time.sleep(4)
                    learner_params = self.param_queue.get()
                    self.sess.run([self.obtain_q_parameters,self.obtain_target_parameters],
                                  feed_dict=self.create_feed_dict(learner_params))

                if self.anealing and self.anealing_steps + self.no_anealing_steps > self.t >= self.no_anealing_steps:
                    self.epsilon -= self.epsilon_step

                self.total_reward += reward_ori
                self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}, session=self.sess))
                self.duration += 1

            elapsed = time.time() - start

            text = 'EPISODE: {0:6d} / ACTOR: {1:3d} / TOTAL_STEPS: {2:8d} / STEPS: {3:5d} / EPSILON: {4:.5f} / TOTAL_REWARD: {5:3.0f} / MAX_Q_AVG: {6:2.4f} / STEPS_PER_SECOND: {7:.1f}'.format(
                self.episode + 1, self.num, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.duration/elapsed)

            print(text)


            with open(self.env_name+'_output.txt','a') as f:
                f.write(text+"\n")

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1


        print("Actor", self.num, "is Over.")
        time.sleep(0.5)

