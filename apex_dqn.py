
import os
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import argparse

from actor import Actor
from leaner import Learner


def actor_work(args, queue, num):#, server):
    sess = tf.InteractiveSession()
    actor = Actor(args, queue, num, sess)
    actor.run()

def leaner_work(args, queue):#, server):
    sess = tf.InteractiveSession()
    leaner = Learner(args, queue, sess)
    leaner.run()


# Train Mode
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_actors', type=int, default=4)
    parser.add_argument('--env_name', type=str, default='Breakout-v0')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--replay_memory_size', type=int, default=200000)
    parser.add_argument('--initial_memory_size', type=int, default=20000, help='Learner waits until RB stores this number of transition.')
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number of episodes each agent plays')
    parser.add_argument('--frame_width', type=int, default=84)
    parser.add_argument('--frame_height', type=int, default=84)
    parser.add_argument('--state_length', type=int, default=4)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)

    args = parser.parse_args()


    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())

    # if args.train:
    #     assert not os.path.exists(args.env_name+'_output.txt'), 'Output file already exists. Change file name.'
    #
    # if not args.load_network:
    #     assert not os.path.exists('saved_networks/'+args.env_name), 'Saved network already exists.'




    if args.train:
        queue = mp.Queue(100)

        # cluster = tf.train.ClusterSpec({'local': ['localhost:2222', 'localhost:2223']})
        # server_learner = tf.train.Server(cluster, job_name='local', task_index=0)

        # with tf.device("/cpu:0"):
        ps = [mp.Process(target=leaner_work, args=(args, queue))]

        for i in range(args.num_actors):
            # server_worker = tf.train.Server(cluster, job_name='local', task_index=i+1)
            ps.append(mp.Process(target=actor_work, args=(args, queue, i)))

        for p in range(len(ps)):
            ps[p].start()
            print(p)

    # Test Mode
    # else:
    #     env = gym.make(ENV_NAME)
    #     env = wrappers.Monitor(env, SAVE_NETWORK_PATH, force=True)
    #     sess = tf.InteractiveSession()
    #     agent = Actor(number=0,sess=sess)
    #     for _ in range(NUM_EPISODES_AT_TEST):
    #         terminal = False
    #         observation = env.reset()
    #         for _ in range(random.randint(1, NO_OP_STEPS)):
    #             last_observation = observation
    #             observation, _, _, _ = env.step(0)  # Do nothing
    #         state = agent.get_initial_state(observation, last_observation)
    #         while not terminal:
    #             last_observation = observation
    #             action = agent.get_action_at_test(state)
    #             observation, _, terminal, _ = env.step(action)
    #             env.render()
    #             processed_observation = agent.preprocess(observation, last_observation)
    #             state =np.append(state[1:, :, :], processed_observation, axis=0)


