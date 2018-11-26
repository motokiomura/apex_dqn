
import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input, Lambda, concatenate
from keras import backend as K
import time


class Memory:
    def __init__(self, size):
        self.transition = deque()
        self.priorities = deque()
        self.size = size
        self.total_p = 0


        self.alpha = 0.6

    def _error_to_priority(self, error_batch):
        priority_batch = []
        for error in error_batch:
            priority_batch.append(error**self.alpha)
        return priority_batch

    def length(self):
        return len(self.transition)

    def add(self, transiton_batch, error_batch):
        priority_batch = self._error_to_priority(error_batch)
        self.total_p += sum(priority_batch)
        self.transition.extend(transiton_batch)
        self.priorities.extend(priority_batch)

    def sample(self, n):
        batch = []
        idx_batch = []
        segment = self.total_p / n

        idx = -1
        sum_p = 0
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            while sum_p < s:
                sum_p += self.priorities[idx]
                idx += 1
            idx_batch.append(idx)
            batch.append(self.transition[idx])
        return batch, idx_batch


    def update(self, idx_batch, error_batch):
        priority_batch = self._error_to_priority(error_batch)
        for i in range(len(idx_batch)):
            change = priority_batch[i] - self.priorities[idx_batch[i]]
            self.total_p += change
            self.priorities[idx_batch[i]] = priority_batch[i]


    def remove(self):
        print("Excess Memory: ", (len(self.priorities) - self.size))
        #[self.transition.popleft() for _ in range(len(self.transition) - NUM_REPLAY_MEMORY)]
        for _ in range(len(self.priorities) - self.size):
            self.transition.popleft()
            p = self.priorities.popleft()
            self.total_p -= p





class Learner:
    def __init__(self,
                 args,
                 queues,
                 sess,
                 target_update_interval=2500,
                 memory_remove_interval=100,
                 batch_size=32,
                 lr=0.00025/4,
                 save_interval=50000,
                 print_interval=30,
                 max_queue_no_added=100
                 ):

        self.env_name = args.env_name
        self.load = args.load
        self.replay_memory_size = args.replay_memory_size
        self.initial_memory_size = args.initial_memory_size
        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.state_length = args.state_length
        # self.n_step = args.n_step
        # self.gamma = args.gamma
        self.gamma_n = args.gamma**args.n_step

        self.queue = queues[0]
        self.param_queue = queues[1]

        self.target_update_interval = target_update_interval
        self.memory_remove_interval = memory_remove_interval
        self.batch_size = batch_size
        self.lr = lr
        self.save_interval = save_interval
        self.print_interval = print_interval
        self.max_queue_no_added = max_queue_no_added
        self.no_added_count = 0

        self.remote_memory = Memory(self.replay_memory_size)

        self.save_network_path = 'saved_networks/' + self.env_name

        self.num_actions = gym.make(args.env_name).action_space.n

        self.t = 0
        self.total_time = 0

        self.queue_not_changed_count = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        self.start = 0

        # with tf.device('/gpu:0'):
        with tf.variable_scope("learner_parameters", reuse=True):
            self.s, self.q_values, q_network = self.build_network()
        self.q_network_weights = self.bubble_sort_parameters(q_network.trainable_weights)

        # Create target network
        with tf.variable_scope("learner_target_parameters", reuse=True):
            self.st, self.target_q_values, target_network = self.build_network()
        self.target_network_weights = self.bubble_sort_parameters(target_network.trainable_weights)

    # Define target network update operation
        self.update_target_network = [self.target_network_weights[i].assign(self.q_network_weights[i]) for i in range(len(self.target_network_weights))]


        # Define loss and gradient update operation
        self.a, self.y, self.error, self.loss, self.grad_update, self.gv, self.cl = self.build_training_op(self.q_network_weights)

        self.sess = sess#tf.InteractiveSession()#server.target)#config=tf.ConfigProto(log_device_placement=True))# gpu_options=tf.GPUOptions(allow_growth=True)))
        #self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        while not self.param_queue.full():
            params = self.sess.run((self.q_network_weights, self.target_network_weights))
            self.param_queue.put(params)
        # print(self.param_queue.get())

        with tf.device("/cpu:0"):
            self.saver = tf.train.Saver(self.q_network_weights)

        # Load network
        if self.load:
            self.load_network()

        if not os.path.exists(self.save_network_path):
            os.makedirs(self.save_network_path)


        # Initialize target network
        self.sess.run(self.update_target_network)


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

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])
        #w = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector. shape=(BATCH_SIZE, num_actions)
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        # shape = (BATCH_SIZE,)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        # error_is = (w / tf.reduce_max(w)) * error
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(self.lr, decay=0.95, epsilon=1.5e-7, centered=True)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=q_network_weights)
        capped_gvs = [(grad if grad is None else tf.clip_by_norm(grad, clip_norm=40), var) for grad, var in grads_and_vars]
        grad_update = optimizer.apply_gradients(capped_gvs)

        #grad_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, error, loss, grad_update ,grads_and_vars, capped_gvs

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.save_network_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')



    def run(self):#, server):



        #print('### len sm',len(self.remote_memory))
        if self.remote_memory.length() < self.initial_memory_size:
            print("Learner Waiting...")

            while not self.queue.empty():
                t_error = self.queue.get()
                self.remote_memory.add(t_error[0], t_error[1])

            while not self.param_queue.full():
                params = self.sess.run((self.q_network_weights, self.target_network_weights))
                self.param_queue.put(params)

            # print('rm len',self.remote_memory.length())
            time.sleep(2)
            return self.run()

        print("Learner Starts!")
        while self.no_added_count < self.max_queue_no_added:
            start = time.time()

            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            terminal_batch = []
            w_batch = []


            if self.queue.empty():
                self.no_added_count += 1
            else:
                self.no_added_count = 0
                while not self.queue.empty():
                    t_error = self.queue.get()
                    self.remote_memory.add(t_error[0], t_error[1])

            while not self.param_queue.full():
                params = self.sess.run((self.q_network_weights, self.target_network_weights))
                self.param_queue.put(params)


            # print('rm len',self.remote_memory.length())
            minibatch, idx_batch = self.remote_memory.sample(self.batch_size)
            #print("mini\n", minibatch[0][0])

            for data in minibatch:
                state_batch.append(data[0])
                action_batch.append(data[1])
                reward_batch.append(data[2])
                #shape = (BATCH_SIZE, 4, 32, 32)
                next_state_batch.append(data[3])
                terminal_batch.append(data[4])

                self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(data[0] / 255.0)]},session=self.sess))

            # Convert True to 1, False to 0
            terminal_batch = np.array(terminal_batch) + 0
            # shape = (BATCH_SIZE, num_actions)
            target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)}, session=self.sess)
            # DDQN
            actions = np.argmax(self.q_values.eval(feed_dict={self.s: np.float32(np.array(next_state_batch) / 255.0)}, session=self.sess), axis=1)
            target_q_values_batch = np.array([target_q_values_batch[i][action] for i, action in enumerate(actions)])
            # shape = (BATCH_SIZE,)
            y_batch = reward_batch + (1 - terminal_batch) * self.gamma_n * target_q_values_batch


            error_batch = self.error.eval(feed_dict={
                self.s: np.float32(np.array(state_batch) / 255.0),
                self.a: action_batch,
                self.y: y_batch
            }, session=self.sess)


            loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
                self.s: np.float32(np.array(state_batch) / 255.0),
                self.a: action_batch,
                self.y: y_batch
                #self.w: w_batch
            })


            self.total_loss += loss
            self.total_time += time.time() - start

            # Memory update
            self.remote_memory.update(idx_batch, error_batch)

            self.t += 1

            if self.t % self.print_interval == 0:
                text_l = 'AVERAGE LOSS: {0:.5F} / AVG_MAX_Q: {1:2.4F} / LEARN PER SECOND: {2:.1F} / NUM LEARN: {3:5d}'.format(
                    self.total_loss/self.print_interval, self.total_q_max/(self.print_interval*self.batch_size), self.print_interval/self.total_time, self.t)
                print(text_l)
                with open(self.env_name+'_output.txt','a') as f:
                    f.write(text_l+"\n")
                #print("Average Loss: ", self.total_loss/PRINT_LOSS_INTERVAL, " / Learn Per Second: ", PRINT_LOSS_INTERVAL/self.total_time, " / AVG_MAX_Q", self.total_q_max/(PRINT_LOSS_INTERVAL*BATCH_SIZE))
                self.total_loss = 0
                self.total_time = 0
                self.total_q_max = 0

            # Remove excess memory
            if self.t % self.memory_remove_interval == 0 and self.remote_memory.length() > self.replay_memory_size:
                self.remote_memory.remove()

            # Update target network
            if self.t % self.target_update_interval == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % self.save_interval == 0:
                save_path = self.saver.save(self.sess, self.save_network_path + '/' + self.env_name, global_step=(self.t))
                print('Successfully saved: ' + save_path)


        print("The Learning is Over.")
        time.sleep(0.5)
