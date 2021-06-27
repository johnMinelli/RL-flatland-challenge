import yaml
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from src.utils.observations import ObserverDAG
from src.utils.utils import Struct
from src.utils.default_observation import normalize_observation
from src.dddqn.train import train
from src.dddqn.eval import eval
from argparse import Namespace

if __name__ == "__main__":
    '''
    Load the configuration to use, and apply the implementation to run
    '''
    with open('../env/env_parameters.yml', 'r') as conf:
        p_env = yaml.load(conf, Loader=yaml.FullLoader)
    p_env = Struct(**p_env)
    with open('../env/training_parameters.yml', 'r') as conf:
        t_env = yaml.load(conf, Loader=yaml.FullLoader)
    t_env = Struct(**t_env)

    # ============== Choose implementation ==============
    # observer, rewards, speed parameters
    # ===================================================
    env = p_env.small_env

    observation_tree_depth = 2
    observation_max_path_depth = 30
    env.observer = ObserverDAG
    # env.observer_params = {"max_depth":observation_tree_depth}
    env.observer_params = {}
    env.observation_normalizer = normalize_observation
    env.predictor = ShortestPathPredictorForRailEnv
    env.predictor_params = {"max_depth":observation_max_path_depth}
    env.observation_radius = 10

    env.speed_profiles = {
        1.: 0.25,
        1. / 2.: 0.25,
        1. / 3.: 0.25,
        1. / 4.: 0.25
    }

    # run
    if t_env.evaluating.active:
        eval(env, t_env)
    else:
        train(env, t_env)

