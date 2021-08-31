import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus

from src.utils import dag_observer
from src.utils.rail_utils import *


def check_if_all_blocked(env):
    """
    Checks whether all the agents are blocked (full deadlock situation).
    In that case it is pointless to keep running inference as no agent will be able to move.
    :param env: current environment
    :return:
    """

    # First build a map of agents in each position
    location_has_agent = {}
    for agent in env.agents:
        if agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and agent.position:
            location_has_agent[tuple(agent.position)] = 1

    # Looks for any agent that can still move
    for handle in env.get_agent_handles():
        agent = env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            continue

        possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
        orientation = agent.direction

        for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
            if possible_transitions[branch_direction]:
                new_position = get_new_position(agent_virtual_position, branch_direction)

                if new_position not in location_has_agent:
                    return False

    # No agent can move at all: full deadlock!
    return True


class DeadlocksController:  #TODO implement default deadlock controller for length
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
        self.deadlocks = [False for _ in range(self.env.get_num_agents())]
        info["deadlocks"] = {}
        for a in range(self.env.get_num_agents()):
            info["deadlocks"][a] = False

        return info


class DeadlocksGraphController:

    def __init__(self, env):
        self.env = env
        self.deadlocks = [False] * self.env.number_of_agents
        self.starvations = [False] * self.env.number_of_agents
        self.starvations_target = [False] * self.env.number_of_agents
        self.deadlock_positions = set()  # for simplicity instead (node to node) this is (x, y, exit_dir)

    """
        Check for new deadlocks, updates info and returns it.
    
        :param info: The information about each agent obs: observation with dict of observation for single handle
        :return: the updated information dictionary
    """
    def check_deadlocks(self, info, obs):
        info["deadlocks"] = {}
        info["starvations"] = {}
        for handle in range(self.env.get_num_agents()):
            info["deadlocks"][handle] = False
            info["starvations"][handle] = False
            graph = obs[handle]
            if not graph is None:
                for label, node in graph.nodes.items():
                    if dag_observer.DagNodeLabel.DEADLOCK in node:
                        opposite_label = (*label[0:2], opposite_dir(label[2]))
                        if node["first_time_detection"]:
                            info["deadlocks"][handle] = True
                            if node['steps_to_deadlock'] == 0:
                                try: self.deadlock_positions.remove(opposite_label)
                                except: pass
                                self.deadlocks[handle] = True
                        elif node['steps_to_deadlock'] == 0:
                            self.deadlock_positions.add(opposite_label)
                            self.deadlocks[handle] = True
                            info["deadlocks"][handle] = True
                            if self.starvations[handle] :
                                self.starvations_target[handle] = True
                        break
                    elif dag_observer.DagNodeLabel.STARVATION in node:
                        self.starvations[handle] = True
                        info["starvations"][handle] = True

        return info

    def reset(self, info):
        self.deadlocks = [False]*len(self.env.agents)
        self.starvations = [False]*len(self.env.agents)
        self.starvations_target = [False]*len(self.env.agents)
        self.deadlock_positions = set()

        info["deadlocks"] = {}
        info["starvations"] = {}
        for a in range(self.env.get_num_agents()):
            info["deadlocks"][a] = False

        return info

    def check_all_blocked(self):
        return np.all(self.deadlocks) or np.all(self.starvations)
