import numpy as np
from flatland.envs.agent_utils import RailAgentStatus


class StatisticsController():
    """
      Store the training statistics.
    """

    def __init__(self, env, env_params):
        self.num_agents = env_params.n_agents
        self.max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim + (env_params.n_agents / env_params.n_cities)))
        self.action_space = env.action_space[0]
        self.action_count = [0] * self.action_space
        self.normalized_score = []
        self.normalized_score_history = []
        self.completion_history = []
        self.completion = 0
        self.episode = 0
        self.score = 0
        self.step = 0

    def reset(self):
        """
        Reset the environment and the statistics
        """
        self.action_count = [0] * self.action_space
        self.normalized_score = []
        self.normalized_score_history = []
        self.completion_history = []
        self.completion = 0
        self.episode = 0
        self.score = 0
        self.step = 0

    def update(self, action_dict, rewards, dones, info):
        """
        Update some statistics and print at the end of the episode
        """
        for a in range(self.num_agents):
            if a in action_dict:
                self.action_count[action_dict[a]] += 1

        self.step += 1

        # Update score and compute total rewards equal to each agent considering the rewards shaped or normal
        self.score += float(sum(rewards.values())) if "original_rewards" not in info \
            else float(sum(info["original_rewards"].values()))

        if dones["__all__"] or self.step >= self.max_steps:
            self._end_episode(info)

    def _end_episode(self, info):

        self.normalized_score = self.score / (self.max_steps * self.num_agents)
        self.normalized_score_history.append(self.normalized_score)
        self.tasks_finished = sum(info["status"][a] in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]
                                  for a in range(self.num_agents))
        self.completion = self.tasks_finished / max(1, self.num_agents)
        self.completion_history.append(self.completion)
        self.action_probs = np.round(self.action_count / np.sum(self.action_count), 3)

        self.action_count = [0] * self.env.action_space[0]
        self.episode += 1
        print(
            "\rEpisode {}"
            "\tScore: {:.3f}"
            " Avg: {:.3f}"
            "\tDone: {:.2f}%"
            " Avg: {:.2f}%"
            "\tAction Probs: {}".format(
                self.episode,
                self.normalized_score,
                np.mean(self.normalized_score_history),
                100 * self.completion,
                100 * np.mean(self.completion_history),
                self._format_action_prob()
            ), end=" ")

    def _format_action_prob(self):
        """
        return a string with the probability linked to each action in the current state
        :return: the writable formatted action probabilities
        """
        actions = ["↻", "←", "↑", "→", "◼"]

        buffer = ""
        for action, action_prob in zip(actions, self.action_probs):
            buffer += action + " " + "{:.3f}".format(action_prob) + " "

        return buffer
