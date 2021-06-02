
class Agent:
    """ Agent abstract class """

    def __init__(self, params=None, state_size=None, action_size=None):
        self.params = params
        self.state_size = state_size
        self.action_size = action_size
        #self.memory = deque(maxlen=5000)

    def act(self, state):
        raise NotImplementedError()

    def step(self, state, action, reward, next_state, done, train=True):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

    def learn(self, experiences):
        raise NotImplementedError()