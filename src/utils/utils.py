from collections import deque
import cv2 as cv # Help us to preprocess the frames
import numpy as np

import configparser as cp
import os

src_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)
base_dir = os.path.join(src_dir, os.path.pardir) 

def preprocess_image(frame, shape=(84, 84)):
    frame = frame.astype(np.uint8)

    #frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY) # convert to grayscale. already did in our vizdoom config
    
    frame = frame[30:-10, 30:-30] # crop the image
    frame = frame /255.0 # normalize

    frame = cv.resize(frame, shape, interpolation=cv.INTER_NEAREST) #resize

    return frame


def stack_frames(stacked_frames, state, reset, stack_size=4):
    frame = preprocess_image(state)

    if reset:
        stacked_frames = deque([frame for i in range(stack_size)], maxlen = stack_size)
    else:
        stacked_frames.append(frame)
    
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

def get_default_config():
    config = cp.ConfigParser()
    config.read_file(open(os.path.join(src_dir, 'default_config.ini')))
    return config
