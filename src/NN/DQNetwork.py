import tensorflow as tf      # Deep Learning library

from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Flatten, Lambda, Add, Subtract

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
