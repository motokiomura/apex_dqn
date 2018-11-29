
import os
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import argparse
import time

from actor import Actor
from learner import Learner


def actor_work(args, queues, num):
    # with tf.device('/cpu:0'):
    sess = tf.InteractiveSession()
    actor = Actor(args, queues, num, sess, param_copy_interval=2000, send_size=200)
    actor.run()

def leaner_work(args, queues):
    # with tf.device('/gpu:0'):
    sess = tf.InteractiveSession()
    leaner = Learner(args, queues, sess, print_interval=3)
    leaner.run()


# Train Mode
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_actors', type=int, default=4)
    parser.add_argument('--env_name', type=str, default='Alien-v0')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--replay_memory_size', type=int, default=2000000)
    parser.add_argument('--initial_memory_size', type=int, default=20000, help='Learner waits until RB stores this number of transition.')
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number of episodes each agent plays')
    parser.add_argument('--frame_width', type=int, default=84)
    parser.add_argument('--frame_height', type=int, default=84)
    parser.add_argument('--state_length', type=int, default=4)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)

    args = parser.parse_args()

    # if args.train:
    #     assert not os.path.exists(args.env_name+'_output.txt'), 'Output file already exists. Change file name.'
    #
    # if not args.load:
    #     assert not os.path.exists('saved_networks/'+args.env_name), 'Saved network already exists.'

    if args.train:
        transition_queue = mp.Queue(100)

        param_queue = 1 #mp.Queue(args.num_actors)

        # with tf.device("/cpu:0"):
        with mp.Manager() as manager:
            param_dict = manager.dict()

            ps = [mp.Process(target=leaner_work, args=(args, (transition_queue, param_dict)))]

            for i in range(args.num_actors):
                ps.append(mp.Process(target=actor_work, args=(args, (transition_queue, param_dict), i)))

            for p in ps:
                p.start()
                time.sleep(0.5)

            for p in ps:
                p.join()

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


