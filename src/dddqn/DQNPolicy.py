from collections import deque
import numpy as np
import tensorflow as tf
import os
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# local imports
from src.common.Policy import Policy
from src.dddqn.model import DQN, DoubleDuelingDQN
from src.dddqn.experience_replay import UniformReplayBuffer, PrioritizedReplay

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DQNPolicy(Policy):
    """ Initializes attributes and constructs CNN model and target_model """

    def __init__(self, state_size, action_size, train_params):
        super(DQNPolicy, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # memory parameters

        self.priority = train_params.replay_buffer.priority
        if not self.priority:
            self.memory = UniformReplayBuffer(train_params.replay_buffer.batch_size,
                                              train_params.replay_buffer.buffer_size)
        else:
            self.memory = PrioritizedReplay(1000, train_params.replay_buffer.batch_size)
        # Hyperparameters
        self.gamma = train_params.dddqn.gamma
        # potremmo separare selection policy in altro file
        self.epsilon = train_params.dddqn.epsilon_start
        self.epsilon_min = train_params.dddqn.epsilon_end
        self.epsilon_decay = train_params.dddqn.epsilon_decay

        self.t_step = train_params.dddqn.t_step
        # 4 time steps for learning, 100 for target network (possible to make it every 100 steps of learning)
        self.update_rate = train_params.dddqn.update_rate
        self.update_network_rate = train_params.dddqn.update_network_rate
        self.tau = train_params.dddqn.tau

        # Construct DQN models
        self.model = DQN(self.action_size)
        self.target_model = DQN(self.action_size)  # fixed q-targets

        # do not specify batch size, only input dimension
        # because batch size varies, the model is used for training
        # where a fixed number of batches is used, but also for
        # action selection, where just a single state is used in forward propagation
        self.model.build(input_shape=(None, self.state_size))
        self.target_model.build(input_shape=(None, self.state_size))



    def act(self, state):
        """Returns actions for given state as per current policy (epsilon-greedy).
                Params
                ======
                    state (array_like): current state
                """
        n = np.random.random()
        if n <= self.epsilon:
            return np.random.randint(0, self.action_size)
        else:

            actions = self.model(state.reshape(1, -1))
            return np.argmax(actions[0])

    def step(self, state, action, reward, next_state, done, train=True):
        # Save experience in replay memory
        if not self.priority:
            self.memory.add((state, action, reward, next_state, done))
        else:
            # q value for current action
            q_pred = self.model(state.reshape(1, -1))[:,action]
            # best q value w.r.t. to actions
            q_new = reward + self.gamma * tf.math.reduce_max(self.target_model(next_state.reshape(1, -1)), axis=1, keepdims=True)

            td_error = tf.math.reduce_sum(abs(q_new - q_pred)).numpy().astype(np.float32)
            self.memory.add((state, action, reward, next_state, done),td_error)



        # Learn every update_rate time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.can_sample() and train:
                if not self.priority:
                    experiences = self.memory.sample()
                    indexes = None
                    weights = None
                else:
                    experiences, indexes, weights = self.memory.sample()
                self.learn(experiences, indexes, weights)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = keras.models.load_model(filename)

    def learn(self, experiences, indexes=None, weights=None):
        """Update value parameters using given batch of experience tuples.
               Params
               ======
                   experiences (Tuple): tuple of (s, a, r, s', done) tuples
                   indexes, weights
        """
        states, actions, rewards, next_states, dones = experiences  # sampled

        # batch of observation in memory is a list, need to turn it to array
        states = np.stack(states, axis=0)
        next_states = np.stack(next_states, axis=0)
        # turn from (batch_size, input_size, 1) to (batch_size, input_size)
        states = states.reshape(states.shape[0], states.shape[1])
        next_states = next_states.reshape(next_states.shape[0], next_states.shape[1])

        q_pred = self.model(states)
        q_next = tf.math.reduce_max(self.target_model(next_states), axis=1, keepdims=True).numpy()
        q_target = np.copy(q_pred)

        for idx, terminal in enumerate(dones):

            q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx]
            if self.priority:
                # update priority of data in replay memory with td error
                self.memory.update(indexes[idx], tf.math.reduce_sum(abs(q_target[idx] - q_pred[idx])).numpy().astype(np.float32))

        # perform gradient descent step and apply importance sampling weight

        # scaler = MinMaxScaler()
        # states = scaler.fit_transform(states)
        # q_target = scaler.fit_transform(q_target)
        scaler = StandardScaler()
        states = scaler.fit_transform(states, q_target)
        q_target = scaler.fit_transform(q_target)
        self.model.train_on_batch(states, q_target, sample_weight=weights)

        if self.t_step % self.update_network_rate == 0:
            self.update_target_model()

    def update_target_model(self):
        new_weights = list(map(lambda weight_tensor: self.tau * weight_tensor + (1.0 - self.tau * weight_tensor), self.model.get_weights()))
        self.target_model.set_weights(new_weights)

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)


class DoubleDuelingDQNPolicy(DQNPolicy):

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

