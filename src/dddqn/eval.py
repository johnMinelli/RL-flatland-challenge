import random
import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from src.env.flatland_railenv import FlatlandRailEnv
from src.utils.log_utils import Timer, TBLogger
from src.dddqn.DQNPolicy import DQNPolicy


def eval(env_params, train_params):
    # Environment parameters
    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_max_path_depth = env_params.observation_max_path_depth
    observation_radius = env_params.observation_radius

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environment
    #env = FlatlandRailEnv(train_params, env_params, tree_observation, env_params.custom_observations, env_params.reward_shaping, train_params.print_stats)
    # for now test without custom observation
    env = FlatlandRailEnv(train_params, env_params, tree_observation,
                          env_params.reward_shaping, train_params.print_stats)
    env.reset()

    # The action space of flatland is 5 discrete actions
    action_size = env.get_rail_env().action_space[0]

    # Max number of steps per episode
    # Official formula used for the evaluation processes [flatland.envs.schedule_generators.sparse_schedule_generator]
    max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim + (env_params.n_agents / env_params.n_cities)))

    policy = DQNPolicy(env.state_size, action_size, train_params)

    print("\nEvaluating: {} agents, {}x{} env, {} episodes.\n".format(env_params.n_agents, env_params.x_dim, env_params.y_dim, train_params.eval_episodes))

    ####################################################################################################################
    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []
    inference_times = []
    preproc_times = []
    agent_times = []
    step_times = []

    for episode_idx in range(train_params.evaluating.episodes):
        seed += 1

        inference_timer = Timer()
        preproc_timer = Timer()
        agent_timer = Timer()
        step_timer = Timer()

        step_timer.start()
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)
        step_timer.end()

        agent_obs = [None] * env.get_num_agents()
        score = 0.0

        final_step = 0
        skipped = 0

        nb_hit = 0
        agent_last_obs = {}
        agent_last_action = {}

        for step in range(max_steps):
            # TODO
            for agent in range(env_params.n_agents):
              # do agent ac
              policy.act()
              # TODO
                # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)
            # TODO
            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

        inference_times.append(inference_timer.get())
        preproc_times.append(preproc_timer.get())
        agent_times.append(agent_timer.get())
        step_times.append(step_timer.get())

        skipped_text = ""
        if skipped > 0:
            skipped_text = "\t⚡ Skipped {}".format(skipped)

        hit_text = ""
        if nb_hit > 0:
            hit_text = "\t⚡ Hit {} ({:.1f}%)".format(nb_hit, (100 * nb_hit) / (n_agents * final_step))

        print(
            "☑️  Score: {:.3f} \tDone: {:.1f}% \tNb steps: {:.3f} "
            "\t🍭 Seed: {}"
            "\t🚉 Env: {:.3f}s  "
            "\t🤖 Policy: {:.3f}s (per step: {:.3f}s) \t[preproc: {:.3f}s \tinfer: {:.3f}s]"
            "{}{}".format(
                normalized_score,
                completion * 100.0,
                final_step,
                seed,
                step_timer.get(),
                agent_timer.get(),
                agent_timer.get() / final_step,
                preproc_timer.get(),
                inference_timer.get(),
                skipped_text,
                hit_text
            )
        )

    results = scores, completions, nb_steps, agent_times, step_times

    ####################################################################################################################

    scores = results[:,0]
    completions = results[:,1]
    nb_steps = results[:,2]
    times = results[:,3]
    step_times = results[:,4]

    print("-" * 200)
    print("✅ Score: {:.3f} \tDone: {:.1f}% \tNb steps: {:.3f} \tPolicy total: {:.3f}s (per step: {:.3f}s)".format(
        np.mean(scores),
        np.mean(completions) * 100.0,
        np.mean(nb_steps),
        np.mean(times),
        np.mean(times) / np.mean(nb_steps)
    ))
    print("⏲️  Policy sum: {:.3f}s \tEnv sum: {:.3f}s \tTotal sum: {:.3f}s".format(
        np.sum(times),
        np.sum(step_times),
        np.sum(times) + np.sum(step_times)
    ))