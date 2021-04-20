from collections import deque
import numpy as np

import matplotlib.pyplot as plt # Display graphs

import argparse
import os

from tensorflow.python import util
import utils.utils as utils
from utils.utils import base_dir, stack_frames

from NN.ActorCriticNetwork import *
import DoomWrapper.DoomEnv as doom


# HYPERPARAMETERS
CURRENT_SCENARIO = 'basic.wad'
CURRENT_CONFIG = 'basic.cfg'

config_path = os.path.join(base_dir, 'config', CURRENT_CONFIG)
scenario_path = os.path.join(base_dir, 'config', CURRENT_SCENARIO)

config = utils.get_default_config()

image_size = tuple(map(int, config['FRAME']['image_size'].split(','))) 
stack_size = int(config['FRAME']['stack_size'])

state_size = (image_size[0], image_size[1], stack_size)
learning_rate = float(config['NN']['learning_rate'])


def train(env, nn_net : ActorCriticNetwork):
    rewards = []
    entropy_term = 0

    all_lengths = []
    average_lengths = []

    episodes_num = int(config['NN']['episodes_num'])
    max_steps = int(config['NN']['max_steps'])
    epsilon = float(config['NN']['epsilon_greedy'])

    stacked_frames = deque([np.zeros((84,84), dtype=np.int) for _ in range(stack_size)], maxlen=stack_size)

    for episode in range(episodes_num):
        log_probs = []
        values = []
        rewards = []

        env.new_episode()
        
        state, stacked_frames = utils.stack_frames(stacked_frames, env.get_state().screen_buffer, True)

        step = 0
        done = False
        while step < max_steps and not done:
            # play episode
            step += 1

            state, stacked_frames = utils.stack_frames(stacked_frames, env.get_state().screen_buffer, False)
            
            if np.random.rand() < epsilon:
                action = nn_net.hot_action[np.random.randint(0, len(nn_net.hot_action))]
            else:
                action = nn_net.get_action(state)

            reward = env.make_action(action.tolist())
            done = env.is_episode_finished()
            rewards.append(reward)

            if done or step == max_steps:
                _, critic_value = nn_net.model.predict(state.reshape((1, *state.shape)))

                critic_value = np.squeeze(critic_value)
                
                total_rewards = np.sum(rewards)
                rewards.append(np.sum(total_rewards))

                if episode % 10 == 0:
                    print(f"episode: {episode}, reward: {total_rewards}\n")
                
        
        # training








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='"test" or "train" or "test_env"')

    args = parser.parse_args()

    if args.mode == 'test_env':
        training = False
        episode_render = True
        print("Testing enviroment with provided configuration.")
        test_game, possible_actions = doom.create_environment(config_path, scenario_path)
        doom.test_environment(test_game, possible_actions)
    
    elif args.mode == 'train':
        training = True
        episode_render = False

        env, possible_actions = doom.create_environment(config_path, scenario_path)
        action_size = len(possible_actions)
        
        actor_critic = ActorCriticNetwork(state_size, action_size, learning_rate)

        train(env, actor_critic)

    elif args.mode == 'test':
        training = False
        episode_render = True
        raise NotImplementedError("Test not yet implemented")
    else:
        raise RuntimeError('Invalid mode specified. Supported are: "test" or "train" or "test_env"')

