import vizdoom as doom       # Doom Environment
import time              
import random

def create_environment(config_path, scenario_path):
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