import yaml
import numpy as np
import networkx as nx
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from spektral.utils import gcn_filter

from src.utils.dag_observer import DagNodeLabel
from src.utils.utils import Struct
from src.dddqn.model import GCN

def get_node_types(nodes):
    attrs = nodes._nodes
    node_types = []

    for index, (label, v) in enumerate(attrs.items()):

        single_node_types = []

        if DagNodeLabel.START in v:
            single_node_types.append('start')
        if DagNodeLabel.TARGET in v:
            single_node_types.append('target')
        if DagNodeLabel.DEADLOCK in v:
            single_node_types.append('deadlock')
        if DagNodeLabel.STARVATION in v:
            single_node_types.append('starvation')
        if DagNodeLabel.CONFLICT in v:
            single_node_types.append('conflict')

        if not len(single_node_types):
            single_node_types.append('other')

        node_types.append('-'.join(single_node_types))


    ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', sparse=False), [0])],  remainder='passthrough')
    encoded_node_types = ct.fit_transform(np.array(node_types, dtype=object).reshape(-1,1))

    return encoded_node_types

def normalize_observation(observation, max_state_size):
    """
    This function normalizes the observation used by the RL algorithm
    """
    # GNN here


    # use dense numpy representation, for this environment
    # a sparse representation is not really needed.
    A = nx.to_numpy_matrix(observation)
    # add identity for self-edges
    A = A + np.eye(A.shape[0])
    A[A > 0] = 1  # adjacency contains weights which we don't want here


    # normalization by degree matrix

    nodes = observation.nodes
    n_nodes = observation.number_of_nodes()
    node_types = get_node_types(nodes)
    n_categories = node_types.shape[1]
    adj_view = observation.adj

    A_hat = gcn_filter(A)
    X = np.zeros([A.shape[0], 1+n_categories], dtype=np.float32)

    for idx, node in enumerate(nodes):

        X[idx, 0] = 0
        X[idx, 1:1+n_categories] = node_types[idx, :]
        adjacent_nodes = adj_view[node]
        weights = [value['weight'] for value in adjacent_nodes.values()]
        X[idx, 0] = min(weights) if len(weights) else 0
    X = X.astype(np.float32)

    gcn = GCN(int(X.shape[1]))

    # [n_nodes, channels]
    observation_normalized = gcn(inputs=[X, A_hat])
    observation_normalized_as_array = observation_normalized.numpy()

    agent_attrs = []
    for node, attrs in nodes._nodes.items():
        if attrs.get('start'):
            agent_attrs.append(attrs.get('velocity', 0))
            agent_attrs.append(attrs.get('nr_malfunctions', 0))
            agent_attrs.append(attrs.get('next_malfunctions', 0))
            agent_attrs.append(attrs.get('malfunction_rate', 0))
            agent_attrs.append(attrs.get('malfunction', 0))
            agent_attrs.append(attrs.get('shortest_path_cost', 0))

    observation_normalized_as_array = np.concatenate(
        (observation_normalized_as_array, np.array(agent_attrs).reshape(-1, 1)))


    n_observations = len(observation_normalized_as_array)
    if n_observations > max_state_size:
        # then need to prune the observation
        nodes_to_prune = n_observations - max_state_size
        # don't want to prune the last elements, the agent attributes
        observation_normalized_as_array = np.concatenate((observation_normalized_as_array[: -6 - nodes_to_prune, :],observation_normalized_as_array[-6:, :]))


    elif n_observations < max_state_size:
        # fill observation array with 0
        observation_normalized_filled = np.zeros((max_state_size,1))
        observation_normalized_filled[0 : n_observations] = observation_normalized_as_array
        return observation_normalized_filled

    return observation_normalized_as_array



