import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from spektral.layers import GCNConv

class DQN(Model):


    def get_config(self):
        pass

    def __init__(self, action_size, dense1_dims=128, dense2_dims=64, loss='huber_loss', learning_rate=1e-2):
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

    def __init__(self, action_size, dense1_dims=128, dense2_dims=64, loss='huber_loss', learning_rate=1e-2):
        super(DoubleDuelingDQN, self).__init__()

        self.layer1 = Dense(dense1_dims, activation='relu', kernel_initializer='he_uniform')
        self.layer2 = Dense(dense2_dims, activation='relu', kernel_initializer='he_uniform')
        self.V = Dense(1, activation=None, kernel_initializer='he_uniform') # scalar state value, raw value
        self.A = Dense(action_size, activation=None, kernel_initializer='he_uniform')


        self.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)

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


class GCN(Model):

    def __init__(self, gcn_dims, n_convolutions=3):

        super().__init__()

        self.graph_convolutions = []
        for n in range(n_convolutions):
            gcn = GCNConv(gcn_dims, activation='relu')
            self.graph_convolutions.append(gcn)
        self.graph_convolutions.append(GCNConv(1, activation='relu'))



    def call(self, inputs, training=None, mask=None):
        X, A_hat = inputs
        out = self.graph_convolutions[0](inputs)
        for graph_conv in self.graph_convolutions[1:]:
            out = graph_conv([out, A_hat])

        return out

if __name__ == '__main__':
    import numpy as np
    x = np.random.rand(4, 231)
    print(x.shape)
    model = DQN(4)
    model.build(input_shape=(None, 231))
    print(model.summary())
    y = model(x)
    print(y.shape)
