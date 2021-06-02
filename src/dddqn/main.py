import yaml
from src.utils import utils
from src.dddqn import train
from src.dddqn import eval
from argparse import Namespace

if __name__ == "__main__":
    '''
    Load the configuration to use, and apply the implementation to run
    '''
    with open('env_parameters.yml', 'r') as conf:
        p_env = yaml.load(conf, Loader=yaml.FullLoader)
    p_env = utils.Struct(**p_env)
    with open('training_parameters.yml', 'r') as conf:
        t_env = yaml.load(conf, Loader=yaml.FullLoader)
    t_env = utils.Struct(**t_env)

    # ============== Choose implementation ==============
    # observer, rewards, speed parameters
    # ===================================================
    env = p_env.small_env

    env["observation_tree_depth"] = 2
    env["observation_radius"] = 10
    env["observation_max_path_depth"] = 30

    env["speed_profiles"] = {
        1.: 0.25,
        1. / 2.: 0.25,
        1. / 3.: 0.25,
        1. / 4.: 0.25}

    # run
    if p_env.training_parameters["evaluation_mode"]:
        eval(Namespace(**p_env.small_env), Namespace(**t_env))
    else:
        train(Namespace(**p_env.small_env), Namespace(**t_env))

