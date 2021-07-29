﻿from copy import deepcopy

import networkx as nx
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum, Grid4Transitions
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.observations import TreeObsForRailEnv
import heapq

import pylab as plt
from src.env.flatland_railenv import FlatlandRailEnv

from src.utils.rail_utils import *


# reason about hen you are in fron of the target    V
# reason about presence of an agent on switch       V
# reason about deadlock on switch                   V
# reason about agent without path to target         V

# when no path to target is admissible:
# primary choice is dead_edge
# secondary choice is dead_end
# remember that the deadlock label will be applied only when two agents TOUCH into an edge
# remember that an edge will be removed from graph only when an agent is in deadlock with 0 step and prev step is switch


class DagObserver(ObservationBuilder):

    def __init__(self, predictor, conflict_radius):
        super().__init__()
        self.predictor = predictor
        self.conflict_radius = conflict_radius
        self.graph = None
        self.observations = dict()
        self.prev_observations = dict()

    def set_env(self, env: FlatlandRailEnv):
        super().set_env(env)
        if self.predictor is not None:
            self.predictor.set_env(self.env)

    def reset(self):
        self._init_graph()
        if self.predictor is not None:
            self.predictor.reset(self.graph)

    def get_many(self, handles=None):
        self.predictions = self.predictor.get_many()

        # compute priority ordered list
        self.prioritized_agents = self._rank_agents()

        # Collect all the different observation for all the agents
        for h in self.prioritized_agents:
            self.observations[h] = self.get(h)
        return self.observations

    def get(self, handle=0):
        di_graph = nx.DiGraph()  # (node, enter_orientation) ----cost, exit_dir---->
        general_graph = deepcopy(self.graph)

        other_agents_position = [(*(a.initial_position if a.position is None else a.position), a.initial_direction if a.position is None else a.direction)
                                 for iter_handle, a in enumerate(self.env.agents) if iter_handle != handle]
        steps = 0
        steps_to_deadlock = 0
        target_flag = False
        deadlock_flag = False

        start_pos = (self.env.agents[handle].initial_position if self.env.agents[handle].position is None else self.env.agents[handle].position)[::-1]
        start_dir = self.env.agents[handle].initial_direction if self.env.agents[handle].direction is None else self.env.agents[handle].direction
        target = self.env.agents[handle].target[::-1]
        node_behind = get_next_pos(*start_pos, opposite_dir(start_dir))
        switch_behind = is_switch(self.env.rail, *node_behind)

        while not is_switch(self.env.rail, *start_pos, start_dir):
            if start_pos == target: target_flag = True; break
            if (*start_pos, start_dir) in other_agents_position: steps_to_deadlock -= 1
            if (*start_pos, opposite_dir(start_dir)) in other_agents_position: deadlock_flag = True; steps_to_deadlock += steps
            if is_dead_end(self.env.rail, *start_pos): start_dir = opposite_dir(start_dir)
            # iterate
            x, y, start_dir = get_next_oriented_pos(self.env.rail, *start_pos, start_dir)
            start_pos = (x, y)
            steps += 1
        di_graph.add_node((*start_pos, start_dir), **{"start": True, **self.encode_start_node_attributes(handle, steps)})

        if target_flag:  # target road
            # di_graph.update(nodes=[((*start_pos, start_dir), {"target": True})])
            # observation = di_graph
            return None

        elif deadlock_flag:  # deadlock road
            if steps_to_deadlock == 0 and switch_behind:  # the edge is full so remove it from the graph
                self._remove_edge_and_transition(self.graph, node_behind[0:2], start_pos, node_behind[2])
            di_graph.update(nodes=[((*start_pos, start_dir), {"deadlock": True, "steps_to_deadlock": steps_to_deadlock })])
            observation = di_graph

        elif steps > 1:  # straight road
            return None

        elif steps == 0:  # switch
            self._copy_graph_structure(di_graph, self.prev_observations[handle])
            observation = di_graph

        else:  # pre-switch
            # add real target in graph
            di_graph.add_node(target, **{"target": True})
            # start exploration and building of DiGraph
            # get switches near agent's target in DiGraph for shortest_path computation
            ending_points = []  # those are cross cells not switch
            for label, node_attr in general_graph.nodes.items():
                if handle in node_attr['targets']:
                    ending_points.append(label)
                    cost, out_dir = node_attr['targets'][handle]
                    for orientation in range(4):
                        if node_attr["trans"][orientation][out_dir] == 1 and is_switch(self.env.rail, *label, orientation):
                            node = (*label, orientation)
                            di_graph.add_node(node)
                            di_graph.add_edge(node, target, **{'weight': cost, 'out_dir': out_dir})

            # if you are in the switch pre target
            if (start_pos in ending_points and ((*start_pos, start_dir), target) in di_graph.edges):
                cost = steps + general_graph.nodes[start_pos]["targets"][handle][0]
                di_graph.update(nodes=[((*start_pos, start_dir), {"shortest_path_cost": cost, "shortest_path": [start_pos]})])
            else:
                if start_pos in ending_points: ending_points.remove(start_pos)
                # -explore target-to-agent without other agents
                shortest_info = self._build_paths_in_directed_graph(deepcopy(general_graph), di_graph, start_pos, start_dir, ending_points, target)
                if shortest_info is None:
                    self._print_graph(self.graph, 'labels.png')
                    di_graph.update(nodes=[((*start_pos, start_dir), {"starvation": True, "shortest_path_cost": 0, "shortest_path": []})])
                    #TODO act of consequence and send the agent in some non disturbing place
                else:
                    di_graph.update(nodes=[((*start_pos, start_dir), {"shortest_path_cost": shortest_info[0], "shortest_path": shortest_info[1]})])
                # -explore path target-to-agent other agents
                if np.any(self.env.dl_controller.deadlocks):
                    for a in np.array(self.env.agents)[[not d for d in self.env.dl_controller.deadlocks]]:
                        if a.status == RailAgentStatus.ACTIVE or a.status == RailAgentStatus.READY_TO_DEPART:
                            # remove busy edges (only opposite direction) from general_graph
                            pos_y, pos_x = a.initial_position if a.position is None else a.position
                            dir = a.initial_direction if a.position is None else a.direction
                            if is_switch(self.env.rail, pos_x, pos_y, dir):
                                for iter_dir in range(4):
                                    t = [o[iter_dir] for o in general_graph.nodes[(pos_x, pos_y)]["trans_node"]]
                                    [self._remove_edge_and_transition(general_graph, (pos_x, pos_y), destination_node, iter_dir) for destination_node in t if destination_node != (0,0)]
                            else:
                                while True:
                                    if is_dead_end(self.env.rail, pos_x, pos_y): dir = opposite_dir(dir)
                                    if is_switch(self.env.rail, pos_x, pos_y, dir): break
                                    pos_x, pos_y, dir = get_next_oriented_pos(self.env.rail, pos_x, pos_y, dir)
                                    steps += 1
                                t = [o[opposite_dir(dir)] for o in general_graph.nodes[(pos_x, pos_y)]["trans_node"]]
                                [self._remove_edge_and_transition(general_graph, (pos_x, pos_y), destination_node, opposite_dir(dir)) for destination_node in t if destination_node != (0,0)]
                    self._build_paths_in_directed_graph(deepcopy(general_graph), di_graph, start_pos, start_dir, ending_points, target)

                # add attributes to nodes based on conflicts
                agent_ranking_pos = self.prioritized_agents[handle]
                for matching_handle in range(agent_ranking_pos):
                    # a conflict node is a common node between the obs DiGraph and the other agent DiGraph (compare only x, y)
                    if self.env.agents[matching_handle].status == RailAgentStatus.ACTIVE and matching_handle in self.prev_observations:
                        matching_graph = self.prev_observations[matching_handle]
                        for start_label, start_node in matching_graph.nodes.items():
                            if "start" in start_node: break
                        node_list = set()
                        [node_list.update(nx.descendants_at_distance(matching_graph, start_label, radius)) for radius in range(self.conflict_radius+1)]
                        possible_conflicts = set()
                        [possible_conflicts.update([(*matching_node[0:2], other_dir) for other_dir in range(4) if len(matching_node) > 2 and other_dir != matching_node[2]]) for matching_node in node_list]
                        for conflict_node in possible_conflicts.intersection(di_graph.nodes.keys()):
                            matched_conflict_node = set([(*conflict_node[0:2],d) for d in range(4)]).intersection(node_list)[0]
                            di_graph.update(nodes=[(conflict_node, {"conflict": True, **self.encode_conflict_node_attributes(matching_handle, start_node, matched_conflict_node)})])

            observation = di_graph

        # # We track what cells where considered while building the observation and make them accessible for rendering
        # visited = set()
        #
        # for _idx in range(10):
        #     # Check if any of the other prediction overlap with agents own predictions
        #     x_coord = self.predictions[handle][_idx][1]
        #     y_coord = self.predictions[handle][_idx][2]
        #
        #     # We add every observed cell to the observation rendering
        #     visited.add((x_coord, y_coord))
        #     if self.predicted_pos[_idx][handle] in np.delete(self.predicted_pos[_idx], handle, 0):
        #         # We detect if another agent is predicting to pass through the same cell at the same predicted time
        #         observation[handle] = 1
        #
        # # This variable will be accessed by the renderer to visualize the observations
        # self.env.dev_obs_dict[handle] = visited

        self.prev_observations[handle] = observation
        return observation

    def _init_graph(self):
        self.graph = nx.MultiDiGraph()
        # create an edge for each pair of connected switch
        visited = np.zeros((self.env.width, self.env.height), np.bool)
        targets = {a.handle: a.target[::-1] for a in self.env.agents}
        start_points = []
        while not np.all(visited[self.env.rail.grid>0]):
            for i, row in enumerate(self.env.rail.grid):
                for j, _ in enumerate(row):
                    if not visited[j, i] and (i,j) not in targets.values():
                        if self.env.rail.grid[j, i] != 0 and is_switch(self.env.rail, i, j):
                            # append cell oriented
                            [start_points.append((i, j, new_dir)) for new_dir, accessible in enumerate(get_allowed_directions(self.env.rail, i, j)) if accessible]; break
                if len(start_points) != 0: break
            while len(start_points) != 0:
                steps = 0
                targets_in_path = {}
                start_x, start_y, start_dir = start_points.pop()
                x, y, dir = get_next_pos(start_x, start_y, start_dir)
                start_edge_exit = dir
                if visited[y, x]: continue
                while True:
                    steps += 1
                    if is_switch(self.env.rail, x, y):
                        if start_x == x and start_y == y: break
                        # update previous node transition
                        self.graph.add_node((start_x, start_y), **self.encode_node_attributes(start_x, start_y, start_dir, steps, targets_in_path, next_node=(x, y)))
                        # add current switch cell to graph with mirrored information
                        self.graph.add_node((x, y), **self.encode_node_attributes(x, y, dir, steps, targets_in_path, prev_node=(start_x, start_y), mirror_direction=True))
                        self.graph.add_edge((start_x, start_y), (x, y), **{'weight': steps,
                                                                   'exitpoint': {(start_x, start_y): start_edge_exit, (x, y): opposite_dir(dir)}, # entering_direction
                                                                   'key': start_edge_exit})  # exit_direction wrt the edge direction
                        self.graph.add_edge((x, y), (start_x, start_y), **{'weight': steps,
                                                                   'exitpoint': {(start_x, start_y): start_edge_exit, (x, y): opposite_dir(dir)},
                                                                   'key': opposite_dir(dir)})
                        # and continue visit in other directions
                        [start_points.append((x, y, new_dir)) for new_dir, accessible in enumerate(get_allowed_directions(self.env.rail, x, y))
                         if accessible and new_dir != opposite_dir(dir)]
                        visited[y, x] = True
                        break
                    elif is_dead_end(self.env.rail, x, y):
                        self.graph.add_node((start_x, start_y), **self.encode_node_attributes(start_x, start_y, start_dir, steps, targets_in_path, dead_end_detected=True))
                    elif visited[y, x]:
                        break
                    else:
                        for handle, target in targets.items():
                            if target == (x,y): targets_in_path[handle] = steps
                        visited[y, x] = True
                        x, y, dir = get_next_oriented_pos(self.env.rail, x, y, dir)

        print(nx.to_dict_of_dicts(self.graph))

    def _build_paths_in_directed_graph(self, exploration_graph, directed_graph, start_pos, start_dir, ending_points, real_target):
        invalid_transitions = []
        shortest_cost, path = self._get_shorthest_path(exploration_graph, invalid_transitions, ending_points, start_pos, start_dir)
        if path is None: return None
        while not self._is_path_in_graph(directed_graph, path, (*start_pos, start_dir), real_target):
            current_node = None
            current_orientation = start_dir
            i = 0
            while i < len(path):
                next_node = path[i]
                if current_node is None: current_node = next_node; i += 1; continue
                node_directed_destinations = exploration_graph.nodes[current_node]["trans_node"][current_orientation]

                if next_node in node_directed_destinations:
                    # transition is valid continue to iterate
                    node_exit_dir = node_directed_destinations.index(next_node)
                    next_orientation = opposite_dir(exploration_graph.get_edge_data(current_node, next_node, node_exit_dir)["exitpoint"][next_node])
                    cost, destination, dead_end_flag = self._update_graph_until_switch(exploration_graph, real_target,
                                                                                       current_node, current_orientation,
                                                                                       next_node, next_orientation)
                    if dead_end_flag: directed_graph.add_node(destination, **{"dead_end": True})
                    directed_graph.add_edge((*current_node, current_orientation), destination, **{'weight': cost, "out_dir": node_exit_dir})
                    # update iterators and continue
                    try:
                        current_node = destination[0:2]
                        current_orientation = destination[2]
                        i = path.index(current_node)+1
                    except:
                        # end of path reached but no allowed direction for target found
                        _, path = self._get_shorthest_path(exploration_graph, invalid_transitions, ending_points, destination[0:2], destination[2])
                        if path is None: break
                        current_node = None
                        current_orientation = destination[2]
                        i = 0
                    continue
                elif current_node in node_directed_destinations:
                    # immediate dead_end from current_node
                    node_exit_dir = node_directed_destinations.index(current_node)
                    cost, destination, _ = self._update_graph_until_switch(exploration_graph, real_target,
                                                                           current_node, current_orientation,
                                                                           current_node, opposite_dir(current_orientation))
                    directed_graph.add_node(destination, **{"dead_end": True})
                    directed_graph.add_edge((*current_node, current_orientation), destination,
                                            **{'weight': cost, "out_dir": node_exit_dir})
                    # update iterators and continue flow
                    current_node = destination[0:2]
                    current_orientation = destination[2]
                    continue
                else:
                    # the edge for this direction is invalid
                    invalid_transitions.append((current_node, current_orientation, next_node))
                    # this case you must have at least a transition possible from your orientation
                    unreachable_node = next_node
                    exploration_node = None
                    next_node = None
                    for node_exit_dir, destination_node in enumerate(node_directed_destinations):
                        if destination_node == (0, 0): continue

                        if unreachable_node in exploration_graph.nodes[current_node]["trans_node"][opposite_dir(node_exit_dir)]:
                            # transition which eventually with a dead and can take you to the unreachable_node
                            next_orientation = opposite_dir(exploration_graph.get_edge_data(current_node, destination_node, node_exit_dir)["exitpoint"][destination_node])
                            cost, destination, dead_end_flag = self._update_graph_until_switch(exploration_graph, real_target,
                                                                                               current_node, current_orientation,
                                                                                               destination_node, next_orientation)
                            if dead_end_flag:
                                # dead end performed making a step
                                directed_graph.add_node(destination, **{"dead_end": True})
                                directed_graph.add_edge((*current_node, current_orientation), destination, **{'weight': cost, "out_dir": node_exit_dir})
                                continue
                            # after one step (switch) ahead search for a dead end
                            destination_node = destination[0:2]
                            next_orientation = destination[2]
                            next_node_directed_destinations = exploration_graph.nodes[destination_node]["trans_node"][next_orientation]
                            if destination_node in next_node_directed_destinations:
                                # (path resolution) immediate dead_end
                                directed_graph.add_edge((*current_node, current_orientation),(*destination_node, next_orientation),
                                                        **{'weight': cost, "out_dir": node_exit_dir})
                                dead_end_cost, dead_end_destination, _ = self._update_graph_until_switch(exploration_graph, real_target,
                                                                                                         destination_node, next_orientation,
                                                                                                         destination_node, opposite_dir(next_orientation))
                                directed_graph.add_node(dead_end_destination, **{"dead_end": True})
                                directed_graph.add_edge((*destination_node, next_orientation), dead_end_destination,
                                                        **{'weight': dead_end_cost, "out_dir": node_directed_destinations.index(destination_node)})
                        elif exploration_node is None: exploration_node = destination_node
                        if next_node is None: next_node = destination_node

                # need a new dijkstra iteration
                next_node = next_node if exploration_node is None else exploration_node
                node_exit_dir = node_directed_destinations.index(next_node)
                next_orientation = opposite_dir(exploration_graph.get_edge_data(current_node, next_node, node_exit_dir)["exitpoint"][next_node])
                cost = exploration_graph.get_edge_data(current_node, next_node, node_exit_dir)["weight"]
                node_exit_dir = node_directed_destinations.index(next_node)
                directed_graph.add_edge((*current_node, current_orientation), (*next_node, next_orientation),
                                        **{'weight': cost, "out_dir": node_exit_dir})
                _, path = self._get_shorthest_path(exploration_graph, invalid_transitions, ending_points, next_node, node_exit_dir)
                if path is None: break
                current_node = None
                current_orientation = next_orientation
                i = 0
            shortest_cost, path = self._get_shorthest_path(exploration_graph, invalid_transitions, ending_points, start_pos, start_dir)
            if path is None: return None
        return shortest_cost, path

    def _is_path_in_graph(self, di_graph, path, start, target):
        undirected_nodes = [(n[0:2]) for n in di_graph.nodes]
        return np.all([node in undirected_nodes for node in path]) and nx.has_path(di_graph, start, target)

    def _update_graph_until_switch(self, general_graph, target, current_node, current_orientation, next_node, next_orientation):
        # find switch
        # update undirected graph with new edges end remove the one not relevant (no switch)
        # update transition matrix of current_node in row current_orientation with real node switch
        start_node_exit_dir = general_graph.nodes[current_node]["trans_node"][current_orientation].index(next_node)
        if current_node == next_node:
            cost = self.graph.nodes[current_node]["dead_end"][current_orientation][start_node_exit_dir]
        else:
            cost = general_graph.get_edge_data(current_node, next_node, start_node_exit_dir)["weight"]
        general_graph.remove_edge(current_node, next_node, key=start_node_exit_dir)

        dead_end = False
        target_found = False
        prev_node = current_node
        prev_orientation = current_orientation
        while not is_switch(self.env.rail, *next_node, next_orientation):
            # if target found and is reachable from your direction stop
            targets = {get_next_pos(*next_node, attr[1], attr[0]): attr for handle, attr in general_graph.nodes[next_node]["targets"].items()}
            if target in targets:
                target_cost, target_exit_dir = targets[target]
                if general_graph.nodes[next_node]["trans"][next_orientation][target_exit_dir] == 1:
                    target_found = 1; break
            next_node_directed_destinations = general_graph.nodes[next_node]["trans_node"][next_orientation]
            for node_exit_dir, destination_node in enumerate(next_node_directed_destinations):
                if destination_node != (0, 0):
                    # remove original edges but leave untouched the intermediary transitions
                    if next_node == destination_node:
                        dead_end = True
                        destination_orientation = opposite_dir(next_orientation)
                        cost += general_graph.nodes[prev_node]["dead_end"][prev_orientation][node_exit_dir]
                    else:
                        destination_orientation = opposite_dir(general_graph.get_edge_data(next_node, destination_node, node_exit_dir)["exitpoint"][destination_node])
                        cost += general_graph.get_edge_data(next_node, destination_node, node_exit_dir)["weight"]
                    # general_graph.remove_edge(next_node, destination_node, key=node_exit_dir)
                    prev_node = next_node
                    prev_orientation = next_orientation
                    next_node = destination_node
                    next_orientation = destination_orientation
                    break
            if current_node == next_node:
                break
        general_graph.add_edge(current_node, next_node, **{'weight': cost,
                                                              'exitpoint': {current_node: start_node_exit_dir, next_node: opposite_dir(next_orientation)},
                                                              'key': start_node_exit_dir})
        general_graph.nodes[current_node]["trans_node"][current_orientation][start_node_exit_dir] = next_node
        if target_found: return cost+target_cost, target, dead_end
        return cost, (*next_node, next_orientation), dead_end

    def _remove_edge_and_transition(self, general_graph, node1, node2, edge_key):
        general_graph.remove_edge(node1, node2, key=edge_key)
        transitions_node = general_graph.nodes[node1]["trans_node"]
        transitions = general_graph.nodes[node1]["trans"]
        for row, destinations in enumerate(transitions_node):
            for col, destination_node in enumerate(destinations):
                if destination_node == node2:
                    transitions_node[row][col] = (0,0)
                    transitions[row][col] = 0
        general_graph.nodes[node1]["trans_node"] = transitions_node
        general_graph.nodes[node1]["trans"] = transitions

    def _get_shorthest_path(self, graph, invalid_transitions, sources, target, allowed_target_dir):
        # shallow copy of the graph: don't modify node attributes
        general_graph = graph.copy()
        reversed_graph = general_graph.reverse()

        for current, orientation, next in invalid_transitions:
            # prev = [prev for prev in [[label for key, edge_data in general_graph[label][current].items() if
            #             edge_data["exitpoint"][current] == opposite_dir(orientation)] for label, a in general_graph.nodes.items() if
            #                 current in general_graph[label]] if len(prev)>0][0][0]
            previous = {edge_data["weight"]:label for label, edges in reversed_graph[current].items() for key, edge_data in edges.items() if edge_data["exitpoint"][current] == opposite_dir(orientation)}
            prev = previous[sorted(previous)[0]]
            cloned_node = (*current, orientation)
            reversed_graph.add_node(cloned_node, **general_graph.nodes[current])

            for end, data in general_graph[prev].items():
                if end == current:
                    [reversed_graph.add_edge(cloned_node, prev, **{**edge_data, 'key': key}) for key, edge_data in data.items()]
                    break
            for end, data in general_graph[current].items():
                if end != next: [reversed_graph.add_edge(end, cloned_node, **{**edge_data, 'key': key}) for key, edge_data in data.items()]
                else:
                    edge_set = deepcopy(data.items())
                    [reversed_graph.remove_edge(next, current, key=key) for key, edge_data in edge_set]
        # remove unfeasible directions
        feasible_destinations = reversed_graph.nodes[target]["trans_node"][allowed_target_dir]
        for destination, attr in general_graph[target].items():
            if not destination in feasible_destinations:
                for edge_key in [a["exitpoint"][target] for i, a in attr.items()]:
                    reversed_graph.remove_edge(destination, target, key=edge_key)
        try:
            cost, path = nx.multi_source_dijkstra(reversed_graph, sources, target)
            path = [node[0:2] for node in path][::-1]
            return cost, path
        except:
            return -1, None

    def _copy_graph_structure(self, new_graph, old_graph):
        new_graph.add_nodes_from(old_graph)
        for label, attr in old_graph.nodes.items():
            if "conflict" in attr: continue
            if "start" in attr:
                new_graph.update(nodes=[(label,
                         {"shortest_path_cost": attr["shortest_path_cost"], "shortest_path": attr["shortest_path"]})])
            else: new_graph.update(nodes=[(label, attr)])
        for label, attr in old_graph.edges.items():
            new_graph.add_edge(*label, **attr)

    def _print_graph(self, graph, name="graph.png"):
        nx.draw(graph, with_labels = True,
            node_color = 'skyblue', node_size = 2200,
            arrowstyle = '->', arrowsize = 20,
            font_size = 10, font_weight = "bold",
            pos = nx.random_layout(graph, seed=13))
        plt.savefig(name)

    def _rank_agents(self):
        list = dict()
        for handle, agent in enumerate(self.env.agents):
            if handle in self.prev_observations.keys():
                start_node = None
                for _, start_node in self.prev_observations[handle].nodes.items():
                    if "start" in start_node: break;
                next_malfunction = agent.malfunction_data["next_malfunction"]
                malfunction = agent.malfunction_data["malfunction"]
                velocity = agent.speed_data["speed"]
                switch = len(start_node["shortest_path"])
                distance = start_node["shortest_path_cost"]
                ratio = ((distance / (next_malfunction + 1 / (malfunction + 1))) / velocity) * switch
                list.update({handle: ratio})
            else:
                list.update({handle: agent.speed_data["speed"]})
        return [a[0] for a in sorted(list.items(), key=lambda x: x[1])]

    def encode_node_attributes(self, x, y, dir, prev_steps, targets, next_node=None, prev_node=None, mirror_direction=False, dead_end_detected=False):
        trans = np.array([[*self.env.rail.get_transitions(y, x, TRANS[dir])] for dir in range(4)])
        trans_node_l = [(0,0)]*4
        trans_node = np.empty(4, dtype=object)
        for r in range(4): trans_node[r] = trans_node_l.copy()
        dead_end = np.zeros((4,4))
        if mirror_direction:
            # per ogni posizione con sbocco in opposite_dir(dir) ha prev_node
            dir = opposite_dir(dir)
            targets_found = {agent: (prev_steps - opposite_side_steps, dir) for agent, opposite_side_steps in targets.items()}
            if not prev_node is None:
                for transitions in trans_node[trans[:, dir] == 1]: transitions[dir] = prev_node
        else:
            targets_found = {agent: (steps, dir) for agent, steps in targets.items()}
            if not next_node is None:
                for transitions in trans_node[trans[:, dir] == 1]: transitions[dir] = next_node
        if dead_end_detected:
            dead_end[trans[:, dir] == 1, dir] = (prev_steps*2)
        # if the node already exist is an update
        if (x,y) in self.graph.nodes:
            old_attr = self.graph.nodes[(x,y)]
            for arry,o in enumerate(trans_node):
                for arrx,t in enumerate(o):
                    if t == (0, 0): trans_node[arry][arrx] = old_attr["trans_node"][arry][arrx]
            if len(old_attr["targets"]) != 0:
                targets_found = {**old_attr["targets"], **targets_found}
        return {"trans": trans, "trans_node": trans_node, "dead_end": dead_end, "targets": targets_found}

    def encode_conflict_node_attributes(self, agent_handle, start_node, conflict):
        agent = self.env.agents[agent_handle]
        conflict_info = dict()
        conflict_info['velocity'] = agent.speed_data['speed']
        conflict_info['conflict_distance'] = nx.shortest_path_length(self.prev_observations[agent_handle], start_node, conflict) + 1
        conflict_info['target_distance'] = start_node['shortest_path_cost']
        conflict_info['nr_malfunctions'] = agent.malfunction_data['nr_malfunctions']
        conflict_info['next_malfunctions'] = agent.malfunction_data['next_malfunctions']
        conflict_info['malfunction_rate'] = agent.malfunction_data['malfunction_rate']
        conflict_info['malfunction'] = agent.malfunction_data['malfunction']
        return {agent: conflict_info}

    def encode_start_node_attributes(self, agent_handle, switch_distance):
        agent = self.env.agents[agent_handle]
        start_info = dict()
        start_info['velocity'] = agent.speed_data['speed']
        start_info['switch_distance'] = switch_distance
        start_info['nr_malfunctions'] = agent.malfunction_data['nr_malfunctions']
        start_info['next_malfunctions'] = agent.malfunction_data['next_malfunction']
        start_info['malfunction_rate'] = agent.malfunction_data['malfunction_rate']
        start_info['malfunction'] = agent.malfunction_data['malfunction']
        return start_info