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
            info["deadlocks"][handle] = self.deadlocks[handle]
            info["starvations"][handle] = self.starvations[handle]
            graph = obs[handle]
            if not graph is None:
                for label, node in graph.nodes.items():
                    if dag_observer.DagNodeLabel.DEADLOCK in node:
                        opposite_label = (*label[0:2], opposite_dir(label[2]))
                        info["deadlocks"][handle] = True
                        if node["first_time_detection"]:
                            if node['steps_to_deadlock'] == 0:
                                try: self.deadlock_positions.remove(opposite_label)
                                except: pass
                                self.deadlocks[handle] = True
                        elif node['steps_to_deadlock'] == 0:
                            self.deadlock_positions.add(opposite_label)
                            self.deadlocks[handle] = True
                            if self.starvations[handle]:
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
