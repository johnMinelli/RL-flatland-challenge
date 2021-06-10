from keras.layers import Dense
from keras import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


class DQN(Model):
    """
    Implements a vanilla deep q network, with simple dense layers
    using huber loss for stability, avoiding exploding gradient ( according to <Human-level control through deep reinforcement
    learning>, Mnih et Al. 2015), by clipping the error term in the Q value update to -1,1 interval
    and He initialization for weights, which is efficient with ReLU (<Delving Deep into Rectifiers:
    Surpassing Human-Level Performance on ImageNet Classification>, Kaiming He et Al.), which solves vanishing and
    exploding gradient by sampling weights from a standard normal distribution and then uses a factor to double the variance
    to address ReLU non linearity

    """

    def get_config(self):
        pass

    def __init__(self, action_size, dense1_dims=24, dense2_dims=12, loss='huber_loss', learning_rate=1e-2):
        super(DQN, self).__init__()

        self.layer1 = Dense(dense1_dims, activation='relu', kernel_initializer='he_uniform')
        self.layer2 = Dense(dense2_dims,  activation='relu', kernel_initializer='he_uniform')
        self.Q = Dense(action_size, activation='relu', kernel_initializer='he_uniform')

        self.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)

    def call(self, state, training=False, mask=None):
        # forward pass, possibly training-specific behavior
        x = self.layer1(state)
        x = self.layer2(x)
        Q = self.Q(x)
        return Q


class DoubleDuelingDQN(Model):

    def __init__(self, action_size, dense1_dims=24, dense2_dims=12, loss='huber_loss', learning_rate=1e-2):
        super(DoubleDuelingDQN, self).__init__()

        self.layer1 = Dense(dense1_dims, activation='relu', kernel_initializer='he_uniform')
        self.layer2 = Dense(dense2_dims, activation='relu', kernel_initializer='he_uniform')
        self.V = Dense(1, activation=None, kernel_initializer='he_uniform') # scalar state value, raw value
        self.A = Dense(action_size, activation=None, kernel_initializer='he_uniform')

        # possibly compile in agent
        self.compile(optimizer=Adam(learning_rate=learning_rate, loss=loss))

    def call(self, state, training=None, mask=None):
        # forward pass has to compute transformation
        # from A and V to Q
        x = self.layer1(state)
        x = self.layer2(x)
        # separate computation of V and A
        V = self.V(x)
        A = self.A(x)
        # mean more stable than max advantage
        Q = V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True))

        return Q

    def advantage(self, state):
        # useful for action selection where we just use advantage stream
        x = self.layer1(state)
        x = self.layer2(x)
        A = self.A(x)

        return A

    def get_config(self):
        pass


if __name__ == '__main__':

    model = DoubleDuelingDQN(4)
    model.build(input_shape=(3,5))
    model.summary()
