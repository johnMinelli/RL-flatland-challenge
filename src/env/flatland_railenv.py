from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool



class FlatlandRailEnv(RailEnv):
    """
    Flatland environment to deal with wrappers.
    """
    def __init__(self, env_params, *args, **kwargs):
        super(FlatlandRailEnv, self).__init__(*args, **kwargs)

        self.params = env_params
        self.env = super()
        self.env_renderer = None
        self.deadlocks_detector = None # TODO
        self.observation_normalizer = None # TODO

    def reset(self):
        obs, info = self.env.reset() # are useful? regenerate_rail=, regenerate_schedule=

        # Reset rendering
        if self.params.render:
            self.env_renderer = RenderTool(self.env, show_debug=True, screen_height=1080, screen_width=1920, gl="PGL")
            self.env_renderer.set_new_rail()

        # Reset custom observations
        # Compute deadlocks
        # Normalization

        return obs, info

    def step(self, action_dict):
        """
        Normalize observations by default, update deadlocks and step.

        :param action_dict:
        :return:
        """
        obs, rewards, dones, info = super().step(action_dict)

        # TODO
        # Compute deadlocks
        # deadlocks = self.deadlocks_detector.step(self.rail_env)
        # info["deadlocks"] = {}
        # for agent in range(self.):
        #     info["deadlocks"][agent] = deadlocks[agent]

        # Normalization
        # for agent in obs:
        #     if obs[agent] is not None:
        #         obs[agent] = self.observation_normalizer.normalize_observation(obs[agent], self.rail_env,
        #                                                                        agent, info["deadlocks"][agent])

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
