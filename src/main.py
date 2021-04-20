import numpy as np

import matplotlib.pyplot as plt # Display graphs

import argparse
import os

from tensorflow.python import util
import utils.utils as utils
from utils.utils import base_dir

from NN.DQNetwork import *
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

        action_size = 3 # env.get_available_buttons_size()
        dqn_net = DQNetwork(state_size, action_size, learning_rate)
        raise NotImplementedError("Training not yet implemented")
    elif args.mode == 'test':
        training = False
        episode_render = True
        raise NotImplementedError("Test not yet implemented")
    else:
        raise RuntimeError('Invalid mode specified. Supported are: "test" or "train" or "test_env"')

