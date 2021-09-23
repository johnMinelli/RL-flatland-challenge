from keras import backend as K
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

from ..common.Policy import Policy
from .model import Actor, Critic

class A2C(Policy):
    # significant sensitivity to layer size
    def __init__(self, state_size, action_size, train_params,  layer1_size=128,
                 layer2_size=64):
        super(A2C, self).__init__()
        # Initialize with 2 different learning rates, one for the actor (the policy)
        # and the critic (the baseline value function). Unlike in DQN, we
        # don't copy weights from one network to another but instead update separately the 2 models.
        self.gamma = train_params.a2c.gamma
        self.alpha = train_params.a2c.alpha
        self.beta = train_params.a2c.beta
        self.state_size = state_size
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.action_size = action_size

        # GCN before
        self.t_step = train_params.a2c.t_step

        self.update_rate = train_params.a2c.update_rate
        # the actor the probability distribution
        # of choosing an action given a certain state
        # the critic tells whether the action is good or not, it approximates the value function
        # by way of the loss function, the actor chooses how to behave, the loss function being a function
        # of the critic value as well
        #self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.actor = Actor(self.action_size, self.alpha)
        self.critic = Critic(None, self.beta)
        #self.policy = Critic(self.action_size, self.beta)

        self.actor.build(input_shape=(None, self.state_size))
        self.critic.build(input_shape=(None, self.state_size))
        #self.policy.build(input_shape=(None, self.state_size))
        self.action_space = [i for i in range(self.action_size)]


    def step(self, state, action, reward, next_state, done, train=True):

        # Learn every update_rate time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if train:
                self.learn((state, action, reward, next_state, done))


    def act(self, observation):
        # adapt observation to state

        probabilities = self.actor.predict(observation.reshape(1, -1))[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def learn(self, experiences):
        # doesn't use memory, so kind of slow?
        state, action, reward, next_state, done = experiences

        state = state.reshape(1, -1)
        next_state = next_state.reshape(1, -1)
        critic_value = self.critic.predict(state)
        next_critic_value = self.critic.predict(next_state)

        # compute target
        target = reward + self.gamma*next_critic_value*(1 - int(done))
        delta = target - critic_value
        self.actor.delta = delta
        # turn action to one-hot-encoding
        actions = np.zeros([1, self.action_size])
        actions[np.arange(1), action] = 1.0

        # both train, actor for choosing the action/policy and
        # critic for informing the actor of the best action to take
        self.actor.train_on_batch(state, actions)
        self.critic.train_on_batch(state, target)