from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizer_v2.rmsprop import RMSprop
from keras import Model


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

    def __init__(self, state_size, action_size, loss='huber_loss', optimizer='Adam'):
        super(DQN, self).__init__()

        self.layer1 = Dense(24, activation='relu', input_shape=(state_size,), kernel_initializer='he_uniform')
        self.layer2 = Dense(12,  activation='relu', kernel_initializer='he_uniform')
        self.layer3 = Dense(action_size, activation='relu', kernel_initializer='he_uniform')

        self.compile(loss=loss, optimizer=optimizer)

    def call(self, inputs, training=False, mask=None):
        # forward pass, possibly training-specific behavior
        out1 = self.layer1(inputs)
        out2 = self.layer2(out1)
        actions = self.layer3(out2)
        return actions


if __name__ == '__main__':

    model = DQN((3,5), 4)
    model.build(input_shape=(3,5))
    model.summary()
