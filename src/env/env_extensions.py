import numpy as np
from flatland.envs.agent_utils import RailAgentStatus


class NormalizerController:
    """
      Wrap the normalization procedures
    """

    def __init__(self, env, env_params):
        self.env = env
        self.max_depth = env_params.observer_params['max_depth']
        self.observation_radius = env_params.observation_radius
        self.observation_normalizer = env_params.observation_normalizer

    def normalize_observations(self, observations):
        for agent in observations:
            if observations[agent]:
                observations[agent] = self.observation_normalizer(observations[agent], self.max_depth, observation_radius=self.observation_radius)
        return observations


class DeadlocksController:
    """
      Wrap the deadlocks procedures
    """

    def __init__(self, env):
        self.env = env
        self.directions = [
            # North
            (-1, 0),
            # East
            (0, 1),
            # South
            (1, 0),
            # West
            (0, -1)]
        self.deadlocks = []

    def check_deadlocks(self, info):
        """
        Check for new deadlocks, updates info and returns it.

        :param info: The information about each agent
        :return: the updated information dictionary
        """
        agents = []
        info["deadlocks"] = {}
        for a in range(self.env.get_num_agents()):
            if self.env.agents[a].status not in [RailAgentStatus.DONE_REMOVED,
                                            RailAgentStatus.READY_TO_DEPART,
                                            RailAgentStatus.DONE]:
                agents.append(a)
                if not self.deadlocks[a]:
                    self.deadlocks[a] = self._check_deadlocks(agents, self.deadlocks)
                if not (self.deadlocks[a]):
                    del agents[-1]
            else:
                self.deadlocks[a] = False

            info["deadlocks"][a] = self.deadlocks[a]

        return info

    def _check_deadlocks(self, a1, deadlocks):
        """
        Recursive procedure to find out whether agents in a1 are in a deadlock.

        :param a1: agents to check
        :param deadlocks: current collections of deadlocked agents
        :param env: railway
        :return: True if a1 is in a deadlock
        """
        a2 = self._check_next_pos(a1[-1])

        # No agents in front
        if a2 is None:
            return False
        # Deadlocked agent in front or loop chain found
        if deadlocks[a2] or a2 in a1:
            return True

        # Investigate further
        a1.append(a2)
        deadlocks[a2] = self._check_deadlocks(a1, deadlocks)

        # If the agent a2 is in deadlock also a1 is
        if deadlocks[a2]:
            return True

        # Back to previous recursive call
        del a1[-1]
        return False

    def _check_next_pos(self, a1):
        """
        Check the next pos and the possible transitions of an agent to find deadlocks.

        :param a1: agent
        :param env: railway
        :return:
        """
        pos_a1 = self.env.agents[a1].position
        dir_a1 = self.env.agents[a1].direction

        if self.env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1)[dir_a1] == 1:
            position_check = (pos_a1[0] + self.directions[dir_a1][0], pos_a1[1] + self.directions[dir_a1][1])
            if not (self.env.cell_free(position_check)):
                for a2 in range(self.env.get_num_agents()):
                    if self.env.agents[a2].position == position_check:
                        return a2
        else:
            return self._check_feasible_transitions(pos_a1, self.env.rail.get_transitions(pos_a1[0], pos_a1[1], dir_a1))

    def _check_feasible_transitions(self, pos_a1, transitions):
        """
        Function used to collect chains of blocked agents.

        :param pos_a1: position of a1
        :param transitions: the transition map
        :param env: the railway
        :return: the agent a2 blocking a1 or None if not present
        """
        for direction, values in enumerate(self.directions):
            if transitions[direction] == 1:
                position_check = (pos_a1[0] + values[0], pos_a1[1] + values[1])
                if not (self.env.cell_free(position_check)):
                    for a2 in range(self.env.get_num_agents()):
                        if self.env.agents[a2].position == position_check:
                            return a2

        return None

    def reset(self, info):
        info["deadlocks"] = {}
        for a in range(self.env.get_num_agents()):
            info["deadlocks"][a] = False

        return info


class StatisticsController:
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
