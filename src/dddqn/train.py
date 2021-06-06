import random
import numpy as np
from flatland.envs.rail_env import RailEnvActions

from src.utils.env_utils import create_rail_env, copy_obs
from src.utils.log_utils import Timer, TBLogger
from src.dddqn.DQNPolicy import DQNPolicy


def train(env_params, train_params):
    # Initialize wandb

    eps_start = train_params.dddqn.epsilon_start

    # Set the seeds
    random.seed(env_params.seed)
    np.random.seed(env_params.seed)

    # Setup the environment
    env = create_rail_env(env_params)
    env.reset()

    # The action space of flatland is 5 discrete actions
    action_size = env.action_space[0]

    # Max number of steps per episode
    # Official formula used for the evaluation processes [flatland.envs.schedule_generators.sparse_schedule_generator]
    max_steps = int(4 * 2 * (env_params.width + env_params.height + (env_params.n_agents / env_params.n_cities)))

    policy = DQNPolicy(env.state_size, action_size, train_params)

    # Timers
    training_timer = Timer()
    step_timer = Timer()
    reset_timer = Timer()
    learn_timer = Timer()

    # TensorBoard writer
    logger = TBLogger(train_params.training.tensorboard_path)

    print("\nTraining: {} agents, {}x{} env, {} episodes.\n".format(env_params.n_agents, env_params.width, env_params.height, train_params.training.episodes))

    ####################################################################################################################
    # Training starts
    training_timer.start()

    agent_prev_obs = [None] * env_params.n_agents
    agent_prev_action = [2] * env_params.n_agents

    for episode in range(train_params.training.episodes+1):
        # do the train execution here
        step_timer.reset()
        reset_timer.reset()
        learn_timer.reset()

        reset_timer.start()
        obs, info = env.reset()
        reset_timer.end()

        # Build initial agent-specific observations
        for agent in range(env_params.n_agents):
            if obs[agent] is not None:
                agent_prev_obs[agent] = copy_obs(obs[agent])

        # Run episode
        for step in range(max_steps):
            # Action dictionary to feed to step
            action_dict = dict()

            # Set used to track agents that didn't skipped the action
            agents_in_action = set()

            for agent in range(env_params.n_agents):

                if info["action_required"][agent]:
                    # If an action is required, the actor predicts an action
                    agents_in_action.add(agent)
                    action_dict[agent] = policy.act(obs[agent])

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = env.step(action_dict)
            step_timer.end()

            # learning step only when the agent has finished or at switch decision
            for agent in range(env_params.n_agents):

                if agent in agents_in_action or done[agent]:
                    learn_timer.start()
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], obs[agent], done[agent])
                    learn_timer.end()

                    agent_prev_obs[agent] = copy_obs(obs[agent])

                    if agent not in action_dict:
                        agent_prev_action[agent] = int(RailEnvActions.DO_NOTHING)
                    else:
                        agent_prev_action[agent] = action_dict[agent]

                if next_obs[agent] is not None:
                    obs[agent] = next_obs[agent]

            if env_params.render:
                env.show_render()

            if done["__all__"]:
                break

        # Epsilon decay
        policy.decay()

        # # Save checkpoints
        # if "checkpoint_interval" in train_params and episode % train_params.checkpoint_interval == 0:
        #     if "save_model_path" in train_params:
        #         policy.save(train_params.save_model_path + "_ep_{}.pt".format(episode)
        #                     if "automatic_name_saving" in train_params and train_params.automatic_name_saving else
        #                     train_params.save_model_path)
        # # Save model and replay buffer at checkpoint
        # if episode % args.training.checkpoint == 0:
        #     policy.save(f'./checkpoints/{training_id}-{episode}')
        #
        #     # Save partial model to wandb
        #     if args.generic.enable_wandb and episode > 0 and episode % args.generic.wandb_checkpoint == 0:
        #         wandb.save(f'./checkpoints/{training_id}-{episode}.local')
        #
        #     # Save replay buffer
        #     if args.replay_buffer.save:
        #         policy.save_replay_buffer(
        #             f'./replay_buffers/{training_id}-{episode}.pkl'
        #         )

        # Rendering
        if env_params.render:
            env.close()

        # Update total time
        training_timer.end()

        if train_params.training.print_stats:
            logger.write(env, train_params.dddqn, {"step": step_timer, "reset": reset_timer, "learn": learn_timer, "train": training_timer}, episode)