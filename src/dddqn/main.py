import yaml
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from src.utils.dag_observer import DagObserver
from src.utils.deadlocks import DeadlocksGraphController
from src.utils.predictors import NullPredictor
from src.utils.utils import Struct
from src.utils.graph_normalizer import normalize_observation
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
    # observer, normalizer, predictor, dl controller
    # ===================================================
    env = p_env.small_env

    env.observer = DagObserver
    env.observer_params = {"conflict_radius": 2}
    env.observation_normalizer = normalize_observation
    env.normalizer_params = {}
    env.predictor = NullPredictor
    env.predictor_params = {}
    env.deadlocks = DeadlocksGraphController

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

