

class ReplayBuffer:

    def __init__(self, batch_size, buffer_size, device=None):
        self.batch_size = batch_size
        self.device = device

    def add(self, experience, td_error=None):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def can_sample(self):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()