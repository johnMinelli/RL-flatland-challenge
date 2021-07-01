import random
import pickle
from collections import deque
import numpy as np

from common.experience_replay import ReplayBuffer


class UniformReplayBuffer(ReplayBuffer):
    '''
    Fixed-size buffer to store experience tuples
    '''

    def __init__(self, batch_size, buffer_size, device=None):
        '''
        Initialize a UniformReplayBuffer object
        '''
        #self.choice_size = choice_size
        super().__init__(batch_size, buffer_size, device)

        self.memory = deque(maxlen=buffer_size)
        assert self.batch_size < buffer_size, "Batch greater than buffer size"
        #self.device = device

    def add(self, experience, td_error=None):
        '''
        Add a new experience to memory
        '''
        self.memory.append(experience)

    def sample(self):
        '''
        Randomly sample a batch of experiences from memory.
        Each returned tensor has shape (batch_size, *)
        '''
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



class SumTree(object):


    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # We are in a binary tree so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

        self.n_entries = 0

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1 # self.capacity -1 to skip every inner node

        """ tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]


    def total_priority(self):
        return self.tree[0]  # Returns the root node

class PrioritizedReplay(ReplayBuffer):
    """
    A version of experience replay which more frequently calls on those experiences of the agent where there is more
    learning value.
    """
    def __init__(self, capacity, batch_size, device=None, per_eps=0.01, per_alpha=0.6, per_beta=0.4,
                 per_beta_increment=0.001):
        """
        :param capacity: memory capacity
        :param per_eps: small constant added to the TD error to ensure that even samples with a low TD error still have
        a small chance of being selected for sampling
        :param per_alpha: a factor used to scale the prioritisation based on the TD error up or down. If per_alpha = 0
        then all of the terms go to 1 and every experience has the same chance of being selected, regardless of the TD
        error. Alternatively, if per_alpha = 0 then “full prioritisation” occurs i.e. every sample is randomly selected
        proportional to its TD error (plus the constant). A commonly used per_alpha value is 0.6 – so that
        prioritisation occurs but it is not absolute prioritisation. This promotes some exploration in addition to the
        PER process
        :param per_beta: the factor that increases the importance sampling weights. A value closer to 1 decreases the
        weights for high priority/probability samples and therefore corrects the bias more
        :param per_beta_increment: the value added to per_beta at each sampling until it is annealed to 1
        """
        assert capacity > batch_size, "Capacity of sumtree needs to be greater than selected batch"
        super().__init__(batch_size, buffer_size=None, device=device)
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.per_eps = per_eps
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment

    def _get_priority(self, td_error):
        """
        :param td_error:
        :return: the priority associated to the passed TD error
        """
        return (np.abs(td_error) + self.per_eps) ** self.per_alpha

    def add(self, experience, td_error=None):
        """
        Update the memory with new experience, the td_error here must be inserted to compute priorities.
        """
        if td_error is None:
            raise Exception("TD Error must be specified when adding experience in Prioritised Experience Replay!")

        self.tree.add(self._get_priority(td_error), experience)

    def sample(self):
        """
        :return: batch, indexes of the data and importance sampling weights
        """
        batch = []
        idxs = []
        segment = self.tree.total_priority() / self.batch_size
        priorities = []

        self.per_beta = np.min([1., self.per_beta + self.per_beta_increment])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            data = 0
            p = None
            idx = None

            while data == 0:
                s = random.uniform(a, b)
                idx, p, data = self.tree.get_leaf(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total_priority()
        # Compute importance sampling weights
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.per_beta)
        # Rescaling weights from 0 to 1
        is_weight /= is_weight.max()
        # so batches are a list of batch_size tuples, each tuple is 5 data elements, state, actions, reward, next, done

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        '''da rivedere
        states = torch.from_numpy(self.v_stack_impr([e.state for e in batch if e is not None])) \
            .float().to(self.device)
        actions = torch.from_numpy(self.v_stack_impr([e.action for e in batch if e is not None])) \
            .long().to(self.device)
        rewards = torch.from_numpy(self.v_stack_impr([e.reward for e in batch if e is not None])) \
            .float().to(self.device)
        next_states = torch.from_numpy(self.v_stack_impr([e.next_state for e in batch if e is not None  ])) \
            .float().to(self.device)
        dones = torch.from_numpy(self.v_stack_impr([e.done for e in batch if e is not None]).astype(np.uint8)) \
            .float().to(self.device)'''

        #return (states, actions, rewards, next_states, dones), idxs, is_weight
        return (states, actions, rewards, next_states, dones)
    """
    def step(self):
        self.beta = np.min([1. - self.e, self.beta + self.beta_increment_per_sampling])
    """

    def update(self, idx, td_error):
        """
        Update priorities of the given indexes with the given error
        :param idx:
        :param td_error:
        """
        self.tree.update(idx, self._get_priority(td_error))

    def can_sample(self):
        return self.__len__() > self.batch_size

    def __len__(self):
        """
        :return: the current size of internal memory.
        """
        return self.tree.n_entries
