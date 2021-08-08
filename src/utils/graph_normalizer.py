
import numpy as np
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from spektral.utils import gcn_filter

from ..dddqn.model import GCN

def get_node_types(nodes):
    attrs = nodes._nodes
    node_types = []
    for label, v in attrs.items():
        if 'start' in v:
            node_types.append('start')
        elif 'target' in v:
            node_types.append('target')
        elif 'deadlock' in v:
            node_types.append('deadlock')
        elif 'starvation' in v:
            node_types.append('starvation')
        elif 'conflict' in v:
            node_types.append('conflict')
        else:
            node_types.append('other')

    ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', sparse=False), [0])],  remainder='passthrough')
    encoded_node_types = ct.fit_transform(np.array(node_types).reshape(-1,1))

    return encoded_node_types

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

    nodes = observation.nodes
    node_types = get_node_types(nodes)
    n_categories = node_types.shape[1]
    adj_view = observation.adj

    A_hat = gcn_filter(A)
    X = np.zeros([A.shape[0], 1+n_categories], dtype=np.float32)

    for idx, node in enumerate(nodes):

        X[idx, 0] = 0
        X[idx, 1:1+n_categories] = node_types[idx, :]
        adjacent_nodes = adj_view[node]
        for v in adjacent_nodes.values():
            X[idx, 0] += v['weight']
    X = X.astype(np.float32)

    gcn = GCN(int(X.shape[1]))
    #gcn.build(input_shape=(None, X.shape[0], X.shape[1]))

    observation_normalized = gcn(inputs=[X, A_hat])
    # [n_nodes, channels]
    print(observation_normalized)
    return observation_normalized



