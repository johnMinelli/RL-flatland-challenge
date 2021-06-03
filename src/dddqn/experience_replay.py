import random
import pickle
from collections import deque


class ReplayBuffer:
    '''
    Fixed-size buffer to store experience tuples
    '''

    def __init__(self, batch_size, buffer_size, device=None):
        '''
        Initialize a ReplayBuffer object
        '''
        #self.choice_size = choice_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        assert self.batch_size < buffer_size, "Batch greater than buffer size"
        #self.device = device

    def add(self, experience):
        '''
        Add a new experience to memory
        '''
        self.memory.append(experience)

    def sample(self):
        '''
        Randomly sample a batch of experiences from memory.
        Each returned tensor has shape (batch_size, *)
        '''
        print()
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.memory, k=self.batch_size)
        )
        # possibly need to convert to numpy array
        return states, actions, rewards, next_states, dones

    def can_sample(self):
        '''
        Check if there are enough samples in the replay buffer
        '''
        return len(self.memory) >= self.batch_size

    def save(self, filename):
        '''
        Save the current replay buffer to a pickle file
        '''
        with open(filename, 'wb') as f:
            pickle.dump(list(self.memory), f)

    def load(self, filename):
        '''
        Load the current replay buffer from the given pickle file
        '''
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)

    def __len__(self):
        '''
        Return the current size of internal memory
        '''
        return len(self.memory)
