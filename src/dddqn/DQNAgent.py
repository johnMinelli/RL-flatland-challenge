from collections import deque
import copy
import numpy as np
import random

# local imports
from dddqn.model import DQN
from common.Agent import Agent


class DQNAgent(Agent):
    """ Initializes attributes and constructs CNN model and target_model """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # memory parameters
        self.buffer_size = 5000
        self.memory = deque(maxlen=self.buffer_size)
        self.batch_size = 512 # minibatch to sample
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        # Hyperparameters
        self.gamma = 1.0  # Discount rate
        # potremmo separare selection policy in altro file
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1  # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.995  # Decay rate for epsilon

        self.t_step = 0
        self.update_rate = 4  # number of steps before learning
        self.update_network_rate = 100 # Number of steps before updating the target network
        self.tau = 1e-3 # update target network

        # Construct DQN models
        self.model = DQN(self.state_size, self.action_size)
        self.target_model = copy.deepcopy(self.model) # fixed q-targets

    def act(self, state):
        """Returns actions for given state as per current policy (epsilon-greedy).
                Params
                ======
                    state (array_like): current state
                """
        actions = self.model.predict(state)

        n = np.random.random()
        print(n)
        if n <= self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return np.argmax(actions[0])

    def step(self, state, action, reward, next_state, done, train=True):
        # Save experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = random.sample(self.memory)
                if train:
                    self.learn(experiences)

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
               Params
               ======
                   experiences (Tuple): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences # sampled

        targets = rewards + (self.gamma * np.amax(self.target_model.predict(next_states)) * (1-dones))

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.t_step % self.update_network_rate == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.tau * self.model.get_weights() + (1.0 - self.tau) * self.target_model.get_weights())
