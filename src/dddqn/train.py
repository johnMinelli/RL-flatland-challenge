import random
import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from src.env.flatland_railenv import FlatlandRailEnv
from src.utils.log_utils import Timer, TBLogger
from src.dddqn.DQNPolicy import DQNPolicy


def train(env_params, train_params):
    # Initialize wandb

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_max_path_depth = env_params.observation_max_path_depth

    # Set the seeds
    random.seed(env_params.seed)
    np.random.seed(env_params.seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observer = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environment
    env = FlatlandRailEnv(train_params, env_params, tree_observer, env_params.custom_observations, env_params.reward_shaping, train_params.print_stats)
    env.reset()

    # The action space of flatland is 5 discrete actions
    action_size = env.get_rail_env().action_space[0]

    # Max number of steps per episode
    # Official formula used for the evaluation processes [flatland.envs.schedule_generators.sparse_schedule_generator]
    max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim + (env_params.n_agents / env_params.n_cities)))

    policy = DQNPolicy(env.state_size, action_size, train_params)  # TODO policy

    # Timers
    training_timer = Timer()
    step_timer = Timer()
    learn_timer = Timer()

    # TensorBoard writer
    logger = TBLogger(train_params.tensorboard_path)

    print("\nTraining: {} agents, {}x{} env, {} episodes.\n".format(env_params.n_agents, env_params.x_dim, env_params.y_dim, train_params.eval_episodes))

    ####################################################################################################################
    # Training starts
    training_timer.start()

    for episode in range(train_params.n_episodes):
        # do the train execution here
        step_timer.reset()
        learn_timer.reset()

        # TODO
