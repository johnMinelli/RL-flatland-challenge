import random
import numpy as np
from flatland.envs.rail_env import RailEnvActions

from src.utils.env_utils import create_rail_env, copy_obs
from src.utils.log_utils import Timer, TBLogger
from src.dddqn.DQNPolicy import DQNPolicy, DoubleDuelingDQNPolicy
from src.dddqn.a2c import A2C
from copy import deepcopy

try:
    import wandb

    use_wandb = True
except ImportError as e:
    print("wandb is not installed, TensorBoard on specified directory will be used!")
    use_wandb = False


def train(env_params, train_params, wandb_config=None):
    # Initialize wandb
    if use_wandb:

        wandb.init(project=train_params.logging.wandb_project,
                   entity=train_params.logging.wandb_entity,
                   tags=train_params.logging.wandb_tag,
                   config=wandb_config,
                   sync_tensorboard=True)

    eps_start = train_params.dddqn.epsilon_start

    # Set the seeds
    random.seed(env_params.seed)
    np.random.seed(env_params.seed)

    # Setup the environment
    env = create_rail_env(env_params)

    # The action space of flatland is 5 discrete actions
    action_size = env.action_space[0]

    # Max number of steps per episode
    # Official formula used for the evaluation processes [flatland.envs.schedule_generators.sparse_schedule_generator]
    max_steps = int(4 * 2 * (env_params.width + env_params.height + (env_params.n_agents / env_params.n_cities)))

    if train_params.training.policy == "dqn":
        policy = DQNPolicy(env.state_size, action_size, train_params)
    elif train_params.training.policy == "dddqn":
        policy = DoubleDuelingDQNPolicy(env.state_size, action_size, train_params)
    elif train_params.training.policy == "a2c":
        policy= A2C(env.state_size, action_size, train_params)


    # Timers
    training_timer = Timer()
    step_timer = Timer()
    reset_timer = Timer()
    learn_timer = Timer()

    # TensorBoard writer
    logger = TBLogger(wandb.run.dir if use_wandb else train_params.training.tensorboard_path)

    print("\nTraining: {} agents, {}x{} env, {} episodes.\n".format(env_params.n_agents, env_params.width, env_params.height, train_params.training.episodes))

    ####################################################################################################################
    # Training starts
    training_timer.start()
    agent_prev_action = [2] * env_params.n_agents


    for episode in range(train_params.training.episodes+1):
        # do the train execution here
        step_timer.reset()
        reset_timer.reset()
        learn_timer.reset()

        reset_timer.start()
        obs, info = env.reset()
        prev_obs = deepcopy(obs)
        reset_timer.end()

        # Run episode
        for step in range(max_steps):
            # Action dictionary to feed to step
            agents_action = dict()

            # Set used to track agents that didn't skipped the action
            agents_policy_guided = set()

            for agent in range(env_params.n_agents):
                if env.dl_controller.deadlocks[agent] or not info['action_required'][agent]:
                    continue
                if info["decision_required"][agent]:
                    # If an action is required, the actor predicts an action
                    agents_policy_guided.add(agent)
                    agents_action[agent] = policy.act(obs[agent])
                else: agents_action[agent] = int(RailEnvActions.MOVE_FORWARD)

            # Environment step
            step_timer.start()
            obs, rewards, done, info = env.step(agents_action)

            step_timer.end()

            for agent in range(env_params.n_agents):

                # learning step only for agents at switch/pre-switch decision
                if info['action_required'][agent] and prev_obs[agent] is not None and obs[agent] is not None:
                    learn_timer.start()
                    policy.step(prev_obs[agent], agent_prev_action[agent], rewards[agent], obs[agent], done[agent])
                    learn_timer.end()

                if agent in agents_action:
                    agent_prev_action[agent] = agents_action[agent]
                else:
                    agent_prev_action[agent] = int(RailEnvActions.DO_NOTHING)

                if obs[agent] is not None:
                    prev_obs[agent] = obs[agent]

            if env_params.render:
                env.show_render()

            if all([done[a.handle] or env.dl_controller.deadlocks[a.handle] or env.dl_controller.starvations[a.handle] for a in env.agents]):
                break

        # Epsilon decay
        if train_params.training.policy != "a2c":
            policy.decay()

        # Save model and replay buffer at checkpoint
        if episode+1 % train_params.training.checkpoints == 0:
            policy.save(f'./checkpoints/training-{episode}')

            # # Save partial model to wandb
            # if args.generic.enable_wandb and episode > 0 and episode % args.generic.wandb_checkpoint == 0:
            #     wandb.save(f'./checkpoints/{training_id}-{episode}.local')

            # # Save replay buffer
            # if args.replay_buffer.save:
            #     policy.save_replay_buffer(
            #         f'./replay_buffers/{training_id}-{episode}.pkl'
            #     )

        # Rendering
        if env_params.render:
            env.close()

        # Update total time
        training_timer.end()


        if train_params.training.print_stats and episode >= 1:

            logger.write(env, train_params.dddqn, {"step": step_timer, "reset": reset_timer, "learn": learn_timer, "train": training_timer}, episode)

        #stopped_for_deadlock = False