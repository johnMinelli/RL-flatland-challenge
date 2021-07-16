import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

from src.env.env_extensions import StatisticsController, NormalizerController, DeadlocksController


class FlatlandRailEnv(RailEnv):
    """
    Flatland environment to deal with wrappers.
    """
    def __init__(self, env_params, *args, **kwargs):
        super(FlatlandRailEnv, self).__init__(*args, **kwargs)

        self.previous_distance = [400] * env_params.n_agents
        self.params = env_params
        self.env_renderer = None
        self.norm_controller = NormalizerController(self, env_params)
        self.dl_controller = DeadlocksController(self)
        self.stats_controller = StatisticsController(self, env_params)

        # Calculate the state size given the depth of the tree observation and the number of features
        n_features_per_node = self.obs_builder.observation_dim
        n_nodes = sum([np.power(4, i) for i in range(self.params.observer_params['max_depth'] + 1)])
        self.state_size = n_features_per_node * n_nodes
        self.stats = {}

    def reset(self):
        obs, info = super().reset()  # are useful? regenerate_rail=, regenerate_schedule=

        # Reset rendering
        if self.params.render:
            self.env_renderer = RenderTool(self.env, show_debug=True, screen_height=1080, screen_width=1920, gl="PGL")
            self.env_renderer.set_new_rail()
        # Normalization phase
        obs = self.norm_controller.normalize_observations(obs)
        # Reset deadlocks
        info = self.dl_controller.reset(info)
        # Reset statistics
        self.stats_controller.reset()

        return obs, info

    def step(self, action_dict):
        """
        Normalize observations by default, update deadlocks and step.

        :param action_dict:
        :return:
        """
        obs, rewards, dones, info = super().step(action_dict)

        # Normalization phase
        obs = self.norm_controller.normalize_observations(obs)
        # Deadlocks check
        info = self.dl_controller.check_deadlocks(info)
        # Rewards progress
        rewards = self._compute_rewards(rewards, info, dones)
        # Stats progress
        stats = self.stats_controller.update(action_dict, rewards, dones, info)
        if stats: self.stats = stats

        return obs, rewards, dones, info

    def show_render(self):
        """
        Open rendering window.

        :return:
        """
        if self.params.render:
            return self.env_renderer.render_env(
                show=True,
                frames=False,
                show_observations=False,
                show_predictions=False)

    def close(self):
        """
        Close rendering window.
        :return:
        """
        if self.params.render:
            return self.env_renderer.close_window()

# private

    def _compute_rewards(self, rewards, info, dones):
        for i_agent, agent in enumerate(self.agents):
            agent_distance_target = np.linalg.norm(np.asarray(agent.position) - np.asarray(agent.target))
            if dones[i_agent]:
                rewards[i_agent] = self.params.rewards.goal_reward
            elif info["deadlocks"][i_agent]:
                rewards[i_agent] = self.params.rewards.deadlock_penalty
            elif agent_distance_target <= self.previous_distance[i_agent]:
                rewards[i_agent] = rewards[i_agent] * self.params.rewards.reducedistance_penalty
            self.previous_distance = agent_distance_target;
        return rewards

# accessors

    def get_rail_env(self):
        return super()
