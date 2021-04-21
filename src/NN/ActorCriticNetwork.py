from collections import deque
import tensorflow as tf      # Deep Learning library
import numpy as np
import utils.utils as utils

from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Flatten, Lambda, Add, Subtract

class ActorCriticNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

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
        pol_act = Dense(action_size, activation='softmax')(pol_act)

        self.model = tf.keras.models.Model(i, (pol_act, val))
        #self.model.compile(tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.Huber())

        self.model.summary()
        self.hot_action = np.identity(action_size, dtype=np.int)

    def get_action(self, state):
        actions_prob, critic_val = self.model.predict(state.reshape((1, *state.shape)))
        actions_prob = np.squeeze(actions_prob)
        idx = np.random.choice(range(self.action_size), p=actions_prob)
        return self.hot_action[idx], actions_prob, critic_val

    def train(self, env, episodes_num, max_steps, beta, gamma):
        rewards = []
        entropy_term = 0

        all_lengths = []
        average_lengths = []

        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for _ in range(self.state_size[2])],
                               maxlen=self.state_size[2])

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

                action, actions_prob, critic_val = self.get_action(state)

                reward = env.make_action(action.tolist())
                values.append(np.squeeze(critic_val))
                done = env.is_episode_finished()
                rewards.append(reward)

                log_prob = np.log(np.dot(actions_prob, action))

                # todo check if entropy is correct.
                entropy = -np.sum(actions_prob * np.log(actions_prob))

                log_probs.append(log_prob)
                entropy_term += entropy

                if done or step == max_steps:
                    _, critic_value = self.model.predict(state.reshape((1, *state.shape)))

                    critic_value = np.squeeze(critic_value)

                    total_rewards = np.sum(rewards)

                    if episode % 10 == 0:
                        print(f"episode: {episode}, reward: {total_rewards}")

            # learn
            with tf.GradientTape() as tape:
                Q_vals = np.zeros(len(values))
                Q_val = 0
                for t in reversed(range(len(rewards))):
                    Q_val = rewards[t] + gamma * Q_val
                    Q_vals[t] = Q_val

                # update actor critic
                advantage = Q_vals - values

                actor_loss = - np.mean(log_probs * advantage)
                critic_loss = 0.5 * np.mean(advantage * advantage)
                total_loss = actor_loss + critic_loss + beta * entropy_term

                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


