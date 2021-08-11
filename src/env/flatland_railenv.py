import numpy as np
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.utils.rendertools import RenderTool

from src.env.env_extensions import StatisticsController, NormalizerController


class FlatlandRailEnv(RailEnv):
    """
    Flatland environment to deal with wrappers.
    """
    def __init__(self, env_params, *args, **kwargs):
        super(FlatlandRailEnv, self).__init__(*args, **kwargs)

        self.params = env_params
        self.env_renderer = None
        self.norm_controller = NormalizerController(self, env_params)
        self.dl_controller = env_params.deadlocks(self)
        self.stats_controller = StatisticsController(self, env_params)

        # Calculate the state size given the depth of the tree observation and the number of features
        self.state_size = self.params.max_state_size
        self.stats = {}
        self.prev_observations = {}

    def reset(self):
        self.obs_builder.set_env(self)
        obs, info = super().reset()  # regenerate_rail=, regenerate_schedule= are useful?

        # Reset rendering
        if self.params.render:
            self.env_renderer = RenderTool(self, show_debug=True, screen_height=1080, screen_width=1920, gl="PGL")
            self.env_renderer.set_new_rail()

        # Encode information for policy action decision
        info = self._encode_info(info, obs)
        # Reset deadlocks
        info = self.dl_controller.reset(info)
        # Normalization phase
        obs = self.norm_controller.normalize_observations(obs)
        # Reset statistics
        self.stats_controller.reset()

        self.prev_observations = obs
        return obs, info

    def step(self, action_dict):
        """
        Normalize observations by default, update deadlocks and step.

        :param action_dict:
        :return:
        """
        obs, rewards, dones, info = super().step(action_dict)
        # Encode information for policy action decision
        info = self._encode_info(info, obs) # TODO change name when it will be defined it's aim
        # Deadlocks check
        info = self.dl_controller.check_deadlocks(info, obs)
        # Normalization phase
        obs = self.norm_controller.normalize_observations(obs)
        # Rewards progress
        rewards = self._compute_rewards(rewards, dones, info)
        # Stats progress
        stats = self.stats_controller.update(action_dict, rewards, dones, info)
        if stats: self.stats = stats


        self.prev_observations = obs
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

    def get_act(self, agent):
        #TODO Verificare che il move forward sia corretto
        return RailEnvActions.MOVE_FORWARD

# private

    def _encode_info(self, info, obs):
        # Use observations to encode information for reward computation
        # When the agent is distant to a switch the observer return None
        # The starvation and pre-touch deadlock are handled in deadlock controller
        info = {**info, "decision_required":{}, "shortest_path":{}, "shortest_path_cost":{}, "shortest_path_pre":{}, "shortest_path_pre_cost":{}}
        for agent in self.agents:
            i_agent = agent.handle
            if obs[i_agent] is None:
                info["decision_required"][i_agent] = False
                info["shortest_path"][i_agent] = []
                info["shortest_path_cost"][i_agent] = 0
                info["shortest_path_pre"][i_agent] = []
                info["shortest_path_pre_cost"][i_agent] = 0
                continue
            info["decision_required"][i_agent] = True
            for _, start_node in obs[i_agent].nodes.items():
                if "start" in start_node: break
            if not "deadlock" in start_node:
                switch = start_node["shortest_path"]
                distance = start_node["shortest_path_cost"]
                info["shortest_path"][i_agent] = switch
                info["shortest_path_cost"][i_agent] = distance
                if not self.prev_observations[i_agent] is None:
                    for _, start_node in self.prev_observations[i_agent].nodes.items():
                        if "start" in start_node: break;
                    switch_pre = start_node["shortest_path"]
                    distance_pre = start_node["shortest_path_cost"]
                else:
                    switch_pre = switch
                    distance_pre = distance
                info["shortest_path_pre"][i_agent] = switch_pre
                info["shortest_path_pre_cost"][i_agent] = distance_pre
        return info

    def _compute_rewards(self, rewards, dones, info):
        # rewards for previous action which caused such observation is given to the policy to learn
        # also will be given prev_step observation and current observation
        for i_agent, agent in enumerate(self.agents):
            if dones[i_agent]:
                rewards[i_agent] = self.params.rewards.goal_reward
            elif info["deadlocks"][i_agent]:
                rewards[i_agent] = self.params.rewards.deadlock_penalty
            elif info["starvations"][i_agent]:
                rewards[i_agent] = self.params.rewards.starvation_penalty
            elif i_agent in info["shortest_path"] and i_agent in info["shortest_path_pre"] and \
                    len(info["shortest_path"][i_agent]) < len(info["shortest_path_pre"][i_agent]):
                rewards[i_agent] = rewards[i_agent] * self.params.rewards.reduce_distance_penalty
        return rewards


# accessors

    def get_rail_env(self):
        return super()
