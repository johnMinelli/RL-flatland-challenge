import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from spektral.layers import GCNConv

class DQN(Model):


    def get_config(self):
        pass

    def __init__(self, action_size, dense_dims=64, dense_layers=2, loss='huber_loss', learning_rate=1e-2):
        super(DQN, self).__init__()

        self.layers_ = []
        for n in range(dense_layers):
            layer = Dense(dense_dims, activation='relu', kernel_initializer='he_uniform')
            self.layers_.append(layer)


        self.layers_.append(Dense(action_size, activation='relu', kernel_initializer='he_uniform'))

        self.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)

    def call(self, state, training=False, mask=None):
        # forward pass, possibly training-specific behavior
        out = self.layers_[0](state)
        for layer in self.layers[1:]:
            out = layer(out)

        return out



class DoubleDuelingDQN(Model):

    def __init__(self, action_size, dense_dims=64, dense_layers=2, loss='huber_loss', learning_rate=1e-2):
        super(DoubleDuelingDQN, self).__init__()

        self.layers_ = []
        for n in range(dense_layers):
            layer = Dense(dense_dims, activation='relu', kernel_initializer='he_uniform')
            self.layers_.append(layer)


        self.V = Dense(1, activation=None, kernel_initializer='he_uniform') # scalar state value, raw value
        self.A = Dense(action_size, activation=None, kernel_initializer='he_uniform')


        self.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)

    def call(self, state, training=None, mask=None):
        # forward pass has to compute transformation
        # from A and V to Q
        out = self.layers_[0](state)
        for layer in self.layers_[1:]:
            out = layer(out)

        # separate computation of V and A
        V = self.V(out)
        A = self.A(out)
        # mean more stable than max advantage
        Q = V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True))

        return Q

    def advantage(self, state):
        # useful for action selection where we just use advantage stream
        out = self.layers_[0](state)
        for layer in self.layers_[1:]:
            out = layer(out)
        A = self.A(out)

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
