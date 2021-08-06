from keras import backend as K
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class Agent(object):
    # significant sensitivity to layer size
    def __init__(self, alpha, beta, gamma = 0.99, n_actions=4, layer1_size=1024,
                 layer2_size=512, input_dims=8):
        # Initialize with 2 different learning rates, one for the actor (the policy)
        # and the critic (the baseline value function). Unlike in DQN, we
        # don't copy weights from one network to another but instead update separately the 2 models.
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions

        # GCN before

        # the actor the probability distribution
        # of choosing an action given a certain state
        # the critic tells whether the action is good or not, it approximates the value function
        # by way of the loss function, the actor chooses how to behave, the loss function being a function
        # of the critic value as well
        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(self.n_actions)]

    # build neural network
    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1]) # related to loss function computation
        # actor critic share 2 dense layers, but then they fork
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        # single action evaluation
        values = Dense(1, activation='linear')

        # closure, how loss functions are handled in Keras
        def custom_loss(y_true, y_pred):
            # y_true being the actual action taken by the agent
            # y_pred the predicted action by the neural network
            # the loss uses the log, so we clip it to an interval that ensures we do not
            # take either 0 or 1
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out) # one-hot representation of the action
            return K.sum(-log_lik*delta) # delta will be calculated in the learning function, related to the output of the critic
        # this is essentially the polocy
        actor = Model(input=[input, delta], output=[probs])
        actor.compile(optimzer=Adam(lr=self.alpha), loss=custom_loss)

        # value function
        critic = Model(input=[input], output=[values])
        critic.compile(optimzer=Adam(lr=self.beta), loss='mean_squared_error')

        # we need a separate policy for choosing the action, while the other is for training
        # choosing an action just entails using a feedforward network, not considering the critic value
        # no need for compiling since we do not use backpropagation
        policy = Model(input=[input], output=[probs])

        return actor, critic, policy

    def choose_action(self, observation):
        # adapt observation to state
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def learn(self, state, action, reward, next_state, done):
        # doesn't use memory, so kind of slow?
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        critic_value = self.critic.predict(state)
        next_critic_value = self.critic.predict(next_state)

        # compute target
        target = reward + self.gamma*next_critic_value*(1 - int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions]) # turn action to one-hot-encoding
        actions[np.arange(1), action] = 1.0

        # both train, actor for choosing the action/policy and
        # critic for informing the actor of the best action to take
        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)