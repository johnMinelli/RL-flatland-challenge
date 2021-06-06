
import copy
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file, sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from src.env.flatland_railenv import FlatlandRailEnv


def create_rail_env(env_params, load_env=""):

    if load_env:
        rail_generator = rail_from_file(load_env)
    else:
        rail_generator = sparse_rail_generator(
            max_num_cities=env_params.n_cities,
            grid_mode=env_params.grid,
            max_rails_between_cities=env_params.max_rails_between_cities,
            max_rails_in_city=env_params.max_rails_in_city,
            seed=env_params.seed
        )

    # Observation builder
    predictor = env_params.predictor(**env_params.predictor_params)
    # tree_observer = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)
    tree_observer = env_params.observer(**env_params.observer_params, predictor=predictor)

    return FlatlandRailEnv(
        env_params,
        env_params.width,
        env_params.height,
        rail_generator = rail_generator,
        schedule_generator = sparse_schedule_generator(env_params.speed_profiles, seed=env_params.seed),
        number_of_agents = env_params.n_agents,
        obs_builder_object = tree_observer,
        malfunction_generator_and_process_data = malfunction_from_params(env_params.malfunction_parameters),
        remove_agents_at_target = True,
        random_seed = env_params.seed
    )


def save_env(path, width, height, num_trains, max_cities,
             max_rails_between_cities, max_rails_in_cities, grid=False, seed=0):
    '''
    Create a RailEnv environment with the given settings and save it as pickle
    '''
    rail_generator = sparse_rail_generator(
        max_num_cities=max_cities,
        seed=seed,
        grid_mode=grid,
        max_rails_between_cities=max_rails_between_cities,
        max_rails_in_city=max_rails_in_cities,
    )
    env = RailEnv(
        width=width,
        height=height,
        rail_generator=rail_generator,
        number_of_agents=num_trains
    )
    env.save(path)


def get_seed(env, seed=None):
    '''
    Exploit the RailEnv to get a random seed
    '''
    seed = env._seed(seed)
    return seed[0]


def copy_obs(obs):
    '''
    Return a deep copy of the given observation
    '''
    if hasattr(obs, "copy"):
        return obs.copy()
    return copy.deepcopy(obs)