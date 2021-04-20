import tensorflow as tf      # Deep Learning library

from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Flatten, Lambda, Add, Subtract

import numpy as np
from tensorflow.python.ops.gen_math_ops import Mul           # Handle matrices
import vizdoom as doom       # Doom Environment

import random                # Handling random number generation
import time                  # Handling time calculation
import cv2 as cv # Help us to preprocess the frames

from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import argparse

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')




# HYPERPARAMETERS
config_path =  r'C:\Users\berna\Documents\Projects\TestGym\Doom\basic.cfg'
scenario_path = r'C:\Users\berna\Documents\Projects\TestGym\Doom\basic.wad'

stack_size = 4 # frane stack dimension 

state_size = (84, 84, stack_size) # K image of size (84x84)
action_size = 3 #game.get_available_buttons_size()
learning_rate = 0.0002

total_episodes = 500
max_steps = 100 #max possible steps in an episode
batch_size = 64

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

gamma = 0.95

pretrain_length = batch_size # 
memory_size = 1000000 # number of experiences the memory can keep

training = True
episode_render = False


def create_environment():
    game = doom.DoomGame()
    
    game.load_config(config_path)
    game.set_doom_scenario_path(scenario_path)

    game.init()

    left = [1,0,0]
    right = [0,1,0]
    shoot = [0,0,1]
    possible_actions = [left, right, shoot]

    return game, possible_actions

def test_environment(env, actions):
    episode_num = 10

    for i in range(episode_num):
        env.new_episode()
        while not env.is_episode_finished():
            state = env.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)

            reward = env.make_action(action)
            print('\treward: ', reward)

            time.sleep(0.02)
        print('Result: ', env.get_total_reward())
        time.sleep(2)
    env.close()


def preprocess_image(frame, shape=(84, 84)):
    frame = frame.astype(np.uint8)

    #frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY) # convert to grayscale. already did in our vizdoom config
    
    frame = frame[30:-10, 30:-30] # crop the image
    frame = frame /255.0 # normalize

    frame = cv.resize(frame, shape, interpolation=cv.INTER_NEAREST) #resize

    return frame


def stack_frames(stacked_frames, state, reset):
    frame = preprocess_image(state)

    if reset:
        stacked_frames = deque([frame for i in range(stack_size)], maxlen = stack_size)
    else:
        stack_frames.append(frame)
    
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        # build inputs
        i = Input(shape=state_size)

        #build CNN
        # 84 x 84 x 4
        x = Conv2D(32, (8, 8), strides=[4,4], activation='relu', padding='valid')(i)
        x = BatchNormalization()(x)

        # 20 x 20 x 32
        x = Conv2D(64, (4, 4), strides=[2,2], activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)

        # 9 x 9 x 64
        x = Conv2D(128, (4, 4), strides=[2,2], activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)

        # 3 x 3 x 128
        x = Flatten()(x)

        # value and policy A2C
        val = Dense(256, activation='relu')(x)
        val = Dense(1)(val)

        pol_act = Dense(256, activation='relu')(x)
        pol_act = Dense(action_size)(pol_act)

        reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))

        q_vals = Add()([val, Subtract()([pol_act, reduce_mean(pol_act)])])
    
        self.model = tf.keras.models.Model(i, q_vals)
        self.model.compile(tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.Huber())

        self.model.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='"test" or "train" or "test_env"')

    args = parser.parse_args()

    if args.mode == 'test_env':
        training = False
        episode_render = True
        print("Testing enviroment with provided configuration.")
        test_game, possible_actions = create_environment()
        test_environment(test_game, possible_actions)
    
    elif args.mode == 'train':
        training = True
        episode_render = False
        dqn_net = DQNetwork(state_size, action_size, learning_rate)
        raise NotImplementedError("Training not yet implemented")
    elif args.mode == 'test':
        training = False
        episode_render = True
        raise NotImplementedError("Test not yet implemented")
    else:
        raise RuntimeError('Invalid mode specified. Supported are: "test" or "train" or "test_env"')

