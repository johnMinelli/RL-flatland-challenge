import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

from src.env.env_extensions import StatisticsController


class FlatlandRailEnv(RailEnv):
    """
    Flatland environment to deal with wrappers.
    """
    def __init__(self, env_params, *args, **kwargs):
        super(FlatlandRailEnv, self).__init__(*args, **kwargs)

        self.params = env_params
        self.env_renderer = None
        self.stats_controller = StatisticsController(self, env_params)
        self.deadlocks_detector = None  # TODO
        self.observation_normalizer = env_params.observer_normalizer  # TODO would be great a wrapper

        # Calculate the state size given the depth of the tree observation and the number of features
        n_features_per_node = self.obs_builder.observation_dim
        n_nodes = sum([np.power(4, i) for i in range(self.params.observer_params['max_depth'] + 1)])
        self.state_size = n_features_per_node * n_nodes

    def reset(self):
        obs, info = super().reset() # are useful? regenerate_rail=, regenerate_schedule=

        # Reset rendering
        if self.params.render:
            self.env_renderer = RenderTool(self.env, show_debug=True, screen_height=1080, screen_width=1920, gl="PGL")
            self.env_renderer.set_new_rail()

        # Reset custom observations
        # Compute deadlocks
        # Normalization phase
        for agent in obs:
            if obs[agent]:
                obs[agent] = self.observation_normalizer(obs[agent], self.params.observer_params['max_depth'], observation_radius=self.params.observation_radius)

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

        # Compute deadlocks
        # deadlocks = self.deadlocks_detector.step(self.rail_env)
        # info["deadlocks"] = {}
        # for agent in range(self.):
        #     info["deadlocks"][agent] = deadlocks[agent]

        # Normalization phase
        for agent in obs:
            if obs[agent]:
                obs[agent] = self.observation_normalizer(obs[agent], self.params.observer_params['max_depth'], observation_radius=self.params.observation_radius)

        self.stats_controller.update(action_dict, rewards, dones, info)

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

    def get_rail_env(self):
        return super()
