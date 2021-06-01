from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizer_v2.rmsprop import RMSprop
from keras import Model


class DQN(Model):

  def __init__(self, state_size, action_size, loss='mse', optimizer='Adam'):
    super(DQN, self).__init__()

    self.layer1 = Conv2D(32, 8, strides=4, activation='relu', input_shape=state_size)
    self.layer2 = Conv2D(64, 4, strides=2, activation='relu')
    self.layer3 = Conv2D(64, 3, strides=1, activation='relu')

    self.layer4 = Flatten()

    self.layer5 = Dense(512, activation="relu"  )
    self.action = Dense(action_size, activation="linear", name='action_layer')

    self.compile(loss=loss, optimizer=optimizer)

  def call(self, inputs, training=False, mask=None):
    # forward pass, possibly training-specific behavior
    # checks on consistency inputs shape
    #assert inputs.shape[1:] == self.state_size, "Input shape isn't consistent"
    out1 = self.layer1(inputs)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    action = self.action(out5)
    return action




if __name__=='__main__':
  # Once the model is created, you can config the model with losses and metrics with model.compile(),
  # train the model with model.fit(), or use the model to do prediction with model.predict().
  model = DQN((512, 512, 1,), 4)
  #
  model.build(input_shape=(1, 512, 512, 1))
  model.summary()
