
import numpy as np
import networkx as nx

from spektral.utils import gcn_filter
from ..dddqn.model import GCN


def normalize_observation(observation):
    """
    This function normalizes the observation used by the RL algorithm
    """
    # GNN here


    # use dense numpy representation, for this environment
    # a sparse representation is not really needed.
    A = nx.to_numpy_matrix(observation)
    # add identity for self-edges
    A = A + np.eye(A.shape[0])
    A[A > 0] = 1 # adjacency contains weights which we don't want here

    # normalization by degree matrix
    A_hat = gcn_filter(A)

    X = np.zeros([A.shape[0], 1], dtype=np.float32)
    nodes = observation.nodes
    adj_view = observation.adj

    for idx, node in enumerate(nodes):

        #X[idx,0] = node[0]
        #X[idx, 1] = node[1]
        X[idx, 0] = 0
        adjacent_nodes = adj_view[node]
        for v in adjacent_nodes.values():
            X[idx, 0] += v['weight']
    X = X.astype(np.float32)

    gcn = GCN(A.shape[0])
    #gcn.build(input_shape=(None, X.shape[0], X.shape[1]))

    observation_normalized = gcn(inputs=[X, A_hat])
    print(observation_normalized.numpy())
    return observation_normalized



