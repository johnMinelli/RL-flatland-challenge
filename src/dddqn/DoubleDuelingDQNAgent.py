import numpy as np
import os
# local imports
from src.common.model import DoubleDuelingDQN
from src.dqn.DQNAgent import DQNAgent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DoubleDuelingDQNAgent(DQNAgent):

    def __init__(self, state_size, action_size, train_params):
        super().__init__(state_size, action_size, train_params)

        self.model = DoubleDuelingDQN(self.action_size)
        self.target_model = DoubleDuelingDQN(self.action_size)

        self.model.build(input_shape=(None, self.state_size))
        self.target_model.build(input_shape=(None, self.state_size))

    def act(self, state):

        n = np.random.random()
        if n <= self.epsilon:
            return np.random.randint(0, self.action_size)
        else:

            # we only use the advantage array for action pick
            actions = self.model.advantage(state.reshape(1, -1))

            return np.argmax(actions[0])

