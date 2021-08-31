from copy import deepcopy

import networkx as nx
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus
import pylab as plt
from src.env.flatland_railenv import FlatlandRailEnv
from src.env.flatland_railenv import showdbg, closedbg

from src.utils.rail_utils import *


#TODO ma se un agent è in malfunction status è corretto continuare a chiamare la get su di lui?
# avrebbe più senso chiamarla solo quando ha finito di malfunzionare. Sappiamo quando questo accade?
#TODO se ci sono due agent in fila che procedono verso il target anche se magari sono allo switch pre-target
# quello dietro cambierà direzione perchè il nodo verrà marcato come conflict


class DagNodeLabel():
    START = "start"
    TARGET = "target"
    DEADLOCK = "deadlock"
    CONFLICT = "conflict"
    STARVATION = "starvation"
    DEAD_END = "dead_end"


class DagObserver(ObservationBuilder):

    def __init__(self, predictor, conflict_radius):
        super().__init__()
        self.predictor = predictor
        self.conflict_radius = conflict_radius
        self.graph = None
        self.observations = dict()
        self.prev_observations = dict()
        self.safety_flag = True

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
        if self.env.dl_controller.deadlocks[handle]: return None
        if self.env.dones[handle]: return None

        di_graph = nx.DiGraph()  # (node, enter_orientation) ----cost, exit_point---->
        general_graph = deepcopy(self.graph)

        active_agents_position = [get_agent_position(a, dir=True) for iter_handle, a in enumerate(self.env.agents) if
                                  iter_handle != handle and self.env.agents[iter_handle].status == RailAgentStatus.ACTIVE or
                                  self.env.agents[iter_handle].status == RailAgentStatus.READY_TO_DEPART]
        dead_agents_position = [get_agent_position(a) for a in np.array(self.env.agents)[self.env.dl_controller.deadlocks]]

        steps = 0
        steps_to_deadlock = 0
        opposite_deviations = 0
        target_flag = False
        deadlock_flag = False

        start_x, start_y, start_dir = get_agent_position(self.env.agents[handle], dir=True)  # (x, y, d)
        start_pos = (start_x, start_y)
        target = self.env.agents[handle].target[::-1]  # (x,y)
        node_behind = get_next_pos(*start_pos, exit_point=access_point_from_dir(start_dir))
        has_switch_behind = False
        if handle in self.prev_observations:
            prev_start_node = self._get_start_node(self.prev_observations[handle])[0]
            if node_behind[0:2] == prev_start_node[0:2]:
                has_switch_behind = True
                node_behind = prev_start_node

        while not is_switch(self.env.rail, *start_pos, start_dir) if not deadlock_flag else any([ is_switch(self.env.rail, *start_pos, d) for d in range(4) if d != start_dir]):
            # update counters and flags
            if start_pos == target: target_flag = True; break
            if (not deadlock_flag) and (*start_pos, start_dir) in active_agents_position and (not start_pos in dead_agents_position):
                steps_to_deadlock -= 1
            if (not deadlock_flag) and steps != 0 and is_switch(self.env.rail, *start_pos) and is_switch(self.env.rail, *start_pos, opposite_dir(np.argmax(general_graph.nodes[start_pos]["trans"][start_dir]))):
                opposite_deviations += 1
            if (not deadlock_flag) and steps != 0 and ( ((*start_pos, opposite_dir(start_dir)) in active_agents_position and opposite_deviations <= 0) or start_pos in dead_agents_position):
                deadlock_flag = True; steps_to_deadlock += steps - 1
            # iterate
            if is_dead_end(self.env.rail, *start_pos) and deadlock_flag: has_switch_behind = True; break
            if is_dead_end(self.env.rail, *start_pos): start_dir = opposite_dir(start_dir)
            x, y, start_dir = get_next_oriented_pos(self.env.rail, *start_pos, start_dir)
            start_pos = (x, y)
            steps += 1
        if (not deadlock_flag) and (start_pos in dead_agents_position): deadlock_flag = True; steps_to_deadlock += steps - 1

        # start_node
        di_graph.add_node((*start_pos, start_dir), **self._encode_start_node_attributes(handle, steps))

        if target_flag:  # target road
            return None
        elif deadlock_flag:  # deadlock road
            if steps_to_deadlock == 0 and has_switch_behind:  # the edge is full so remove it from the graph
                self._remove_edge_and_transition(self.graph, node_behind[0:2], start_pos, node_behind[2])
            di_graph.update(nodes=[((*start_pos, start_dir), self._encode_dl_node_attributes(steps_to_deadlock, has_switch_behind))])
            return di_graph

        elif np.array_equal(general_graph.nodes[start_pos]["trans"][start_dir], [0,0,0,0]) or \
            all([any([(n[0],n[1], opp_d) in active_agents_position for opp_d in range(4) if opp_d != n[2]]) for n in
                 [get_next_pos(*start_pos, d) for d,v in enumerate(general_graph.nodes[start_pos]["trans"][start_dir]) if v == 1]]):
            # switch already dead (no dest) or all possible dest are in other active_agents_positions (all busy)
            for label, edges in self.graph.reverse()[start_pos].items():  # remove all directions from the graph
                for key, edge_data in edges.items():
                    self._remove_edge_and_transition(self.graph, label, start_pos, key)
            # trigger deadlock for agent
            di_graph.update(nodes=[((*start_pos, start_dir), self._encode_dl_node_attributes(dl_steps=0, first_time_detection=True))])
            return di_graph

        elif steps > 1:  # straight road
            return None

        elif steps == 0 and self._get_start_node(self.prev_observations[handle]) == [(*start_pos, start_dir)]:  # switch
            # reuse previous observation
            self._copy_graph_structure(di_graph, self.prev_observations[handle])
            observation = di_graph

        else:  # pre-switch
            # start exploration and building of DiGraph
            # get switches near agent's target in DiGraph for shortest_path computation
            starvation_flag = self.env.dl_controller.starvations[handle]
            target, ending_points = self._build_target_in_directed_graph(general_graph, di_graph, target, starvation_flag)
            # if you are in the switch pre target
            if start_pos in ending_points and ((*start_pos, start_dir), target) in di_graph.edges:
                cost = steps + general_graph.nodes[start_pos]["targets"][handle][0]
                di_graph.update(nodes=[((*start_pos, start_dir), {"shortest_path_cost": cost, "shortest_path": [start_pos]})])
            else:
                if start_pos in ending_points:
                    #TESTME probably never ever called
                    ending_points.remove(start_pos)
                # -explore target-to-agent without other agents
                shortest_info = self._build_paths_in_directed_graph(deepcopy(general_graph), di_graph, start_pos, start_dir, ending_points, target)
                if shortest_info is None:
                    di_graph.update(nodes=[((*start_pos, start_dir), {DagNodeLabel.STARVATION: True})])
                    # no path available for target: send him to a deadlock edge
                    if (not np.array_equal(ending_points, [(x,y) for x,y,_ in self.env.dl_controller.deadlock_positions])) and len(self.env.dl_controller.deadlock_positions) != 0:
                        target, ending_points = self._build_target_in_directed_graph(general_graph, di_graph, None, starvation_flag=True)  # here a starvation case
                        shortest_info = self._build_paths_in_directed_graph(deepcopy(general_graph), di_graph, start_pos, start_dir, ending_points, target)
                if shortest_info is None: shortest_info = (0,[])
                di_graph.update(nodes=[((*start_pos, start_dir), {"shortest_path_cost": shortest_info[0], "shortest_path": shortest_info[1]})])

                # -explore path target-to-agent other agents
                non_blocked_agents = [not d if h != handle else False for h, d in enumerate(self.env.dl_controller.deadlocks)]
                if np.any(non_blocked_agents):
                    for a in np.array(self.env.agents)[non_blocked_agents]:
                        # remove busy edges (only opposite direction) from general_graph
                        pos_x, pos_y, dir = get_agent_position(a, dir=True)
                        if is_switch(self.env.rail, pos_x, pos_y, dir):
                            for iter_dir in range(4):
                                t = [o[iter_dir] for o in general_graph.nodes[(pos_x, pos_y)]["trans_node"]]
                                [self._remove_edge_and_transition(general_graph, (pos_x, pos_y), destination_node, iter_dir, ending_points) for destination_node in  set(t) if destination_node != (0,0)]
                        else:
                            while not is_switch(self.env.rail, pos_x, pos_y, dir):
                                if is_dead_end(self.env.rail, pos_x, pos_y): dir = opposite_dir(dir)
                                pos_x, pos_y, dir = get_next_oriented_pos(self.env.rail, pos_x, pos_y, dir)
                            t = [o[access_point_from_dir(dir)] for o in general_graph.nodes[(pos_x, pos_y)]["trans_node"]]
                            [self._remove_edge_and_transition(general_graph, (pos_x, pos_y), destination_node, access_point_from_dir(dir), ending_points) for destination_node in set(t) if destination_node != (0, 0)]
                    self._build_paths_in_directed_graph(deepcopy(general_graph), di_graph, start_pos, start_dir, ending_points, target,2)

                # add attributes to nodes based on conflicts
                agent_ranking_pos = self.prioritized_agents.index(handle)
                for matching_handle in self.prioritized_agents[:agent_ranking_pos]:
                    # find conflict nodes: common nodes between the obs DiGraph of the agents (compare only x, y)
                    if self.env.agents[matching_handle].status == RailAgentStatus.ACTIVE and matching_handle in self.prev_observations:
                        matching_graph = self.prev_observations[matching_handle]
                        start_label, start_node = self._get_start_node(matching_graph)
                        node_list = set()
                        [node_list.update(nx.descendants_at_distance(matching_graph, start_label, radius)) for radius in range(self.conflict_radius+1)]
                        possible_conflicts = set()
                        [possible_conflicts.update([(*matching_node[0:2], other_dir) for other_dir in range(4) if len(matching_node) > 2 and other_dir != matching_node[2]]) for matching_node in node_list]
                        for conflict_node in possible_conflicts.intersection(di_graph.nodes.keys()):
                            matched_conflict_node = list(set([(*conflict_node[0:2],d) for d in range(4)]).intersection(node_list))[0]
                            di_graph.update(nodes=[(conflict_node, self._encode_conflict_node_attributes(matching_handle, (start_label, start_node), matched_conflict_node))])

            observation = di_graph

        self.prev_observations[handle] = observation
        return observation

    def _init_graph(self):

        def check_transition_complete(x, y):
            for directions, destinations, dead_ends in zip(self.graph.nodes[(x, y)]["trans"], self.graph.nodes[(x, y)]["trans_node"], self.graph.nodes[(x, y)]["dead_end"]):
                for direction, destination, dead_end in zip(directions, destinations, dead_ends):
                    if direction != 0 and destination == (0, 0) and dead_end == 0:
                        return False
            return True

        self.graph = nx.MultiDiGraph()
        # create an edge for each pair of connected switch
        visited = np.zeros((self.env.width, self.env.height), np.bool)
        targets = {a.handle: a.target[::-1] for a in self.env.agents}
        start_points = []
        j = 0

        while (not np.all(visited[self.env.rail.grid>0])) or len(start_points) != 0:
            while len(start_points) == 0:
                row = self.env.rail.grid[j]
                i = 0
                while len(row) > i:
                    if not visited[j, i] and (i, j) not in targets.values():
                        if self.env.rail.grid[j, i] != 0 and is_switch(self.env.rail, i, j):
                            # append cell with orientation
                            [start_points.append((i, j, new_dir)) for new_dir, is_accessible in enumerate(get_allowed_points(self.env.rail, i, j)) if is_accessible]; break
                    i += 1
                if len(self.env.rail.grid)-1 > j: j += 1
                else: j = 0
            while len(start_points) != 0:
                steps = 0
                targets_in_path = {}
                start_x, start_y, start_dir = start_points.pop(0)
                x, y, dir = get_next_pos(start_x, start_y, start_dir)
                start_access_point = dir

                if self.graph.has_node((start_x, start_y)):
                    visited[start_y, start_x] = check_transition_complete(start_x, start_y)
                    if visited[start_y, start_x]: continue

                while True:
                    steps += 1
                    if is_switch(self.env.rail, x, y):
                        if start_x == x and start_y == y: break
                        # update previous node transition
                        self.graph.add_node((start_x, start_y), **self._encode_node_attributes(start_x, start_y, start_dir, steps, targets_in_path, next_node=(x, y)))
                        # add current switch cell to graph with mirrored information
                        self.graph.add_node((x, y), **self._encode_node_attributes(x, y, dir, steps, targets_in_path, prev_node=(start_x, start_y), mirror_direction=True))
                        self.graph.add_edge((start_x, start_y), (x, y), **{'weight': steps,
                               'access_point': {(start_x, start_y): start_access_point, (x, y): access_point_from_dir(dir)},  # entering_direction
                               'key': start_access_point})  # exit_direction wrt the edge direction
                        self.graph.add_edge((x, y), (start_x, start_y), **{'weight': steps,
                               'access_point': {(start_x, start_y): start_access_point, (x, y): access_point_from_dir(dir)},
                               'key': access_point_from_dir(dir)})
                        visited[start_y, start_x] = check_transition_complete(start_x, start_y)

                        # and continue visit in other directions
                        [start_points.append((x, y, new_dir)) for new_dir, is_accessible in enumerate(get_allowed_points(self.env.rail, x, y))
                                if is_accessible and new_dir != opposite_dir(dir)]
                        visited[y, x] = True
                        break
                    elif is_dead_end(self.env.rail, x, y):
                        self.graph.add_node((start_x, start_y), **self._encode_node_attributes(start_x, start_y, start_dir, steps, targets_in_path, dead_end_detected=True))
                    else:
                        for handle, target in targets.items():
                            if target == (x, y): targets_in_path[handle] = (steps, target)
                        visited[y, x] = True
                        x, y, dir = get_next_oriented_pos(self.env.rail, x, y, dir)

        # print("DAG:", nx.to_dict_of_dicts(self.graph))

    def _build_paths_in_directed_graph(self, exploration_graph, directed_graph, start_pos, start_dir, ending_points, target, ignore=1):
        quit = None; invalid_transitions = []
        shortest_cost, path = self._get_shorthest_path(exploration_graph, invalid_transitions, ending_points, start_pos, start_dir)
        if path is None: return None
        while not self._is_path_in_graph(directed_graph, path, (*start_pos, start_dir), target):
            current_node = None
            current_orientation = start_dir
            i = 0
            while i < len(path):
                next_node = path[i]
                if current_node is None: current_node = next_node; i += 1; continue
                node_direct_destinations = exploration_graph.nodes[current_node]["trans_node"][current_orientation]

                if next_node in node_direct_destinations:
                    # transition is valid: continue on the path
                    indices = [i for i, n in enumerate(node_direct_destinations) if n == next_node and ((current_node), i, next_node) not in invalid_transitions]
                    destinations = {}
                    for current_exit_point in indices:
                        next_orientation = dir_from_access_point(exploration_graph.get_edge_data(current_node, next_node, current_exit_point)['access_point'][next_node])
                        cost, destination, dead_end_flag, ending_point_flag = self._update_graph_until_switch(exploration_graph, ending_points,
                                                                                       current_node, current_orientation,
                                                                                       next_node, next_orientation)
                        if ending_point_flag: destination = target
                        if dead_end_flag: directed_graph.add_node(destination, **{DagNodeLabel.DEAD_END: True})
                        directed_graph.add_edge((*current_node, current_orientation), destination, **{'weight': cost, 'exit_point': current_exit_point})
                        if ending_point_flag: break
                        if destination[0:2] in destinations: destinations[destination[0:2]].append(destination[2])
                        else: destinations[destination[0:2]] = [destination[2]]

                    if ending_point_flag: break
                    for node_destination, orientations_destination in destinations.items():
                        if len(orientations_destination)>1:
                            #unify the transitions for the multiple (both) orientations
                            new_attr = exploration_graph.nodes[node_destination]
                            all_transitions_destinations = new_attr["trans_node"]
                            all_transitions = new_attr["trans"]
                            new_t = [0,0,0,0]
                            new_td = [(0,0),(0,0),(0,0),(0,0)]
                            for orientation, (transition, transition_destinations) in enumerate(zip(all_transitions, all_transitions_destinations)):
                                if orientation in orientations_destination:
                                    new_t = [valid if valid!=0 else default for default, valid in zip(new_t, transition)]
                                    new_td = [valid if valid!=(0,0) else default for default, valid in zip(new_td, transition_destinations)]
                            for orientation in orientations_destination:
                                new_attr["trans"][orientation] = new_t
                                new_attr["trans_node"][orientation] = new_td
                            exploration_graph.update(nodes=[(node_destination, new_attr)])

                    # update iterators and continue
                    current_node = list(destinations.keys())[0]
                    current_orientation = destinations[current_node][0]
                    try:  i = i+path[i:].index(current_node)+1
                    except: # end of path reached and no allowed direction for target found
                        _, path = self._get_shorthest_path(exploration_graph, invalid_transitions, ending_points, current_node, current_orientation)
                        if path is None: break
                        current_node = None  #orientation already valorized to destination[2]
                        i = 0
                    continue
                elif current_node in node_direct_destinations:
                    # immediate dead_end from current_node
                    current_exit_point = node_direct_destinations.index(current_node)
                    cost, destination, _, ending_point_flag = self._update_graph_until_switch(exploration_graph, ending_points,
                                                                           current_node, current_orientation,
                                                                           current_node, opposite_dir(current_orientation))
                    if ending_point_flag: destination = target
                    directed_graph.add_node(destination, **{DagNodeLabel.DEAD_END: True})
                    directed_graph.add_edge((*current_node, current_orientation), destination, **{'weight': cost, 'exit_point': current_exit_point})
                    if ending_point_flag: break
                    # update iterators and continue flow
                    current_node = destination[0:2]
                    current_orientation = destination[2]
                    continue
                else:
                    # this node transition for this direction is invalid
                    if (current_node, current_orientation, next_node) not in invalid_transitions:
                        invalid_transitions.append((current_node, current_orientation, next_node))
                    elif (current_node, current_orientation, next_node) == quit:
                        quit = True; break
                        #TESTME test if it still happen
                    else: quit = (current_node, current_orientation, next_node)
                    # this case you must have at least a transition possible from your orientation
                    unreachable_node = next_node
                    exploration_node = None
                    next_node = None
                    for current_exit_point, destination_node in enumerate(node_direct_destinations):
                        if destination_node == (0, 0): continue

                        if unreachable_node in exploration_graph.nodes[current_node]["trans_node"][dir_from_access_point(current_exit_point)]:
                            # transition which eventually with a dead and can take you to the unreachable_node
                            next_orientation = opposite_dir(exploration_graph.get_edge_data(current_node, destination_node, current_exit_point)['access_point'][destination_node])
                            cost, destination, dead_end_flag, _ = self._update_graph_until_switch(exploration_graph, ending_points,
                                                                                               current_node, current_orientation,
                                                                                               destination_node, next_orientation)
                            if dead_end_flag:
                                # dead end performed making a step
                                directed_graph.add_node(destination, **{DagNodeLabel.DEAD_END: True})
                                directed_graph.add_edge((*current_node, current_orientation), destination, **{'weight': cost, 'exit_point': current_exit_point})
                                continue
                            # after one step (switch) ahead search for a dead end
                            destination_node = destination[0:2]
                            next_orientation = destination[2]
                            next_node_direct_destinations = exploration_graph.nodes[destination_node]["trans_node"][next_orientation]
                            if destination_node in next_node_direct_destinations:
                                # (path resolution) immediate dead_end
                                directed_graph.add_edge((*current_node, current_orientation),(*destination_node, next_orientation),
                                                        **{'weight': cost, 'exit_point': current_exit_point})
                                dead_end_cost, dead_end_destination, _, ending_point_flag = self._update_graph_until_switch(exploration_graph, ending_points,
                                                                                                         destination_node, next_orientation,
                                                                                                         destination_node, opposite_dir(next_orientation))
                                if ending_point_flag: dead_end_destination = target  # then continue and maybe find a better path
                                directed_graph.add_node(dead_end_destination, **{"dead_end": True})
                                directed_graph.add_edge((*destination_node, next_orientation), dead_end_destination,
                                                        **{'weight': dead_end_cost, 'exit_point': node_direct_destinations.index(destination_node)})
                        elif exploration_node is None: exploration_node = destination_node
                        if next_node is None: next_node = destination_node

                # need a new dijkstra iteration
                next_node = next_node if exploration_node is None else exploration_node
                current_exit_point = node_direct_destinations.index(next_node)
                next_orientation = dir_from_access_point(exploration_graph.get_edge_data(current_node, next_node, current_exit_point)['access_point'][next_node])
                cost = exploration_graph.get_edge_data(current_node, next_node, current_exit_point)["weight"]
                directed_graph.add_edge((*current_node, current_orientation), (*next_node, next_orientation),
                                        **{'weight': cost, 'exit_point': current_exit_point})
                _, path = self._get_shorthest_path(exploration_graph, invalid_transitions, ending_points, next_node, next_orientation)
                if path is None: break
                current_node = None
                current_orientation = next_orientation
                i = 0
            shortest_cost, path = self._get_shorthest_path(exploration_graph, invalid_transitions, ending_points, start_pos, start_dir)
            if path is None:
                return None
            if quit == True: break
        return shortest_cost, path

    def _update_graph_until_switch(self, general_graph, ending_points, current_node, current_orientation, next_node, next_orientation):
        # find switch
        # update general_graph with new edges end remove the ones in the path not involving a switch
        # update transition matrix of current_node in row current_orientation with real node switch
        start_node_exit_point = general_graph.nodes[current_node]["trans_node"][current_orientation].index(next_node)
        if current_node == next_node:
            cost = self.graph.nodes[current_node]["dead_end"][current_orientation][start_node_exit_point]
        else:
            cost = general_graph.get_edge_data(current_node, next_node, start_node_exit_point)["weight"]
            general_graph.remove_edge(current_node, next_node, key=start_node_exit_point)

        dead_end_found = False
        target_found = False
        prev_node = current_node
        prev_orientation = current_orientation
        while not is_switch(self.env.rail, *next_node, next_orientation):
            # if ending_point found, stop
            if next_node in ending_points: target_found = True; break
            next_node_direct_destinations = general_graph.nodes[next_node]["trans_node"][next_orientation]
            for next_exit_point, destination_node in enumerate(next_node_direct_destinations):
                if destination_node != (0, 0):
                    # remove original edges but leave untouched the intermediary transitions
                    if next_node == destination_node:
                        dead_end_found = True
                        destination_orientation = opposite_dir(next_orientation)  # ok
                        cost += general_graph.nodes[next_node]["dead_end"][next_orientation][next_exit_point]  # not ok
                    else:
                        destination_orientation = dir_from_access_point(general_graph.get_edge_data(next_node, destination_node, next_exit_point)['access_point'][destination_node])
                        cost += general_graph.get_edge_data(next_node, destination_node, next_exit_point)["weight"]
                    # general_graph.remove_edge(next_node, destination_node, key=next_exit_point)
                    prev_node = next_node
                    prev_orientation = next_orientation
                    next_node = destination_node
                    next_orientation = destination_orientation
                    break
            if current_node == next_node:
                break
        general_graph.add_edge(current_node, next_node, **{'weight': cost,
                'access_point': {current_node: start_node_exit_point, next_node: access_point_from_dir(next_orientation)},
                'key': start_node_exit_point})
        new_attr = general_graph.nodes[current_node]
        prev_node = new_attr["trans_node"][current_orientation][start_node_exit_point]
        for orientiations in new_attr["trans_node"]:
            if orientiations[start_node_exit_point] == prev_node:
                orientiations[start_node_exit_point] = next_node
        general_graph.update(nodes=[(current_node, new_attr)])
        return cost, (*next_node, next_orientation), dead_end_found, target_found

    def _remove_edge_and_transition(self, general_graph, node1, node2, edge_key, ending_points=[]):
        if not self.safety_flag:
            general_graph.remove_edge(node1, node2, key=edge_key)
        else:
            try:
                general_graph.remove_edge(node1, node2, key=edge_key)
            except:
                pass
        exit_point = edge_key
        nodes_to_propagate_action = []
        all_transitions_destinations = general_graph.nodes[node1]["trans_node"]
        all_transitions = general_graph.nodes[node1]["trans"]
        for orientation, destinations in enumerate(all_transitions_destinations):
            if destinations[exit_point] == node2:
                all_transitions_destinations[orientation][exit_point] = (0, 0)
                all_transitions[orientation][exit_point] = 0
                if np.array_equal(all_transitions[orientation], [0, 0, 0, 0]):
                    nodes_affected = [(label, key) for label, edges in general_graph.reverse()[node1].items() for
                                      key, edge_data in
                                      edges.items() if
                                      edge_data['access_point'][node1] == access_point_from_dir(orientation) and
                                      ((len(ending_points) == 0 and len(
                                          general_graph.nodes[label]["targets"]) == 0) or (label, key) not in ending_points)]
                    nodes_to_propagate_action += nodes_affected
        for n, k in nodes_to_propagate_action:
            self._remove_edge_and_transition(general_graph, n, node1, k, ending_points)

    def _get_shorthest_path(self, graph, invalid_transitions, sources, target, allowed_target_dir):  # TODO graph direct
        # shallow copy of the graph: don't modify node attributes
        general_graph = graph.copy()
        reversed_graph = general_graph.reverse()
        for current, orientation, next in invalid_transitions[::-1]:
            previous = [label[0:2] for label, edges in reversed_graph[current].items() for key, edge_data in edges.items() if edge_data['access_point'][current] == opposite_dir(orientation)]
            cloned_node = (*current, 1)
            if cloned_node not in reversed_graph.nodes:  # since you can have multiple invalid_trans on the same current
                reversed_graph.add_node(cloned_node, **general_graph.nodes[current])
                for end, data in reversed_graph[current].items():
                    if not end[0:2] in previous and (current,orientation,end[0:2]) not in invalid_transitions:
                        [reversed_graph.add_edge(cloned_node, end, **{**edge_data, 'key': key}) for key, edge_data in data.items()]
            else:
                for end, data in deepcopy(reversed_graph[cloned_node].items()):
                    if end[0:2] in previous:
                        edge_set = deepcopy(data.items())
                        [reversed_graph.remove_edge(cloned_node, end, key=key) for key, edge_data in edge_set]
            for end, data in deepcopy(general_graph[current].items()):
                if end[0:2] == next[0:2]:
                    [reversed_graph.add_edge(next, cloned_node, **{**edge_data, 'key': key}) for key, edge_data in data.items()]
                    edge_set = deepcopy(data.items())
                    for key, edge_data in edge_set:
                        try: reversed_graph.remove_edge(next, current, key=key)
                        except: pass
                        [reversed_graph.remove_edge((*next, clone), current, key=key) for clone in range(4) if reversed_graph.get_edge_data((*next, clone), current, key)]
        # remove unfeasible directions
        feasible_destinations = reversed_graph.nodes[target]["trans_node"][allowed_target_dir]
        for destination, attr in general_graph[target].items():
            if not destination in feasible_destinations:
                for edge_key in [a['access_point'][target] for i, a in attr.items()]:
                    try: reversed_graph.remove_edge(destination, target, key=edge_key)
                    except: pass
        # convert to a standard DiGraph using the cheapest edge weight
        n = (-1,-1); w = 0; edge_set = deepcopy(reversed_graph.edges.items())
        for l,a in edge_set:
            if l[0:2]==n[0:2]:
                if a["weight"] < w:
                    w = a["weight"]
                    reversed_graph.remove_edge(*n)
                else: reversed_graph.remove_edge(*l)
            else: n=l
        reversed_graph = nx.DiGraph(reversed_graph)
        valid_paths = []
        for i, s in enumerate(sources):
            try: valid_paths.append(nx.shortest_simple_paths(reversed_graph, s, target).__next__())
            except: pass
        if len(valid_paths)!=0:
            paths_costs = [nx.path_weight(reversed_graph, p, "weight") for p in valid_paths]
            minimum_shortest_simple_path = [node[0:2] for node in valid_paths[np.argmin(paths_costs)]][::-1]
            return min(paths_costs), minimum_shortest_simple_path
        else: return -1, None

    def _rank_agents(self):  # less is better
        list = dict()
        for handle, agent in enumerate(self.env.agents):
            if handle in self.prev_observations.keys():
                _, start_node = self._get_start_node(self.prev_observations[handle])
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

    def _build_target_in_directed_graph(self, general_graph, directed_graph, real_target, starvation_flag):
        # if agent is starving use a deadlock edge as new target
        # get switches near agent's target and return them with the target itself
        ending_points = []  # those are cross cells not switch
        if starvation_flag:
            fake_target = real_target
            # with that you suppose that if you end in starvation you can only develop a deadlock
            dying_positions = self.env.dl_controller.deadlock_positions
            fake_target = real_target
            # for pos_x, pos_y, exit_point in dying_positions:
            if len(dying_positions) != 0:
                pos_x, pos_y, exit_point = list(dying_positions)[0]
                label = (pos_x, pos_y)
                ending_points.append(label)
                for orientation in range(4):
                    if self.graph.nodes[label]["trans"][orientation][exit_point] == 1 and is_switch(self.env.rail, *label, orientation):
                        node = (*label, orientation)
                        directed_graph.add_node(node)
                        fake_target = get_next_pos(*label, exit_point, 1)[0:2]  # add fake target in graph
                        directed_graph.add_edge(node, fake_target, **{'weight': 1, 'exit_point': exit_point})
            return fake_target, ending_points
        else:
            # add real target in graph
            directed_graph.add_node(real_target, **{"target": DagNodeLabel.TARGET})
            for label, node_attr in general_graph.nodes.items():
                for handle, target_attr in node_attr['targets'].items():  # targets = {agent: (steps, exit_point, (target_label))}
                    if real_target in target_attr:
                        ending_points.append(label)
                        cost, exit_point = target_attr[0:2]
                        for orientation in range(4):
                            if node_attr["trans"][orientation][exit_point] == 1 and is_switch(self.env.rail, *label, orientation):
                                node = (*label, orientation)
                                directed_graph.add_node(node)
                                directed_graph.add_edge(node, real_target, **{'weight': cost, 'exit_point': exit_point})
                        break
            return real_target, ending_points

    # Node Utils

    def _encode_node_attributes(self, x, y, dir, prev_steps, targets, next_node=None, prev_node=None, mirror_direction=False, dead_end_detected=False):
        trans = np.array([[*self.env.rail.get_transitions(y, x, TRANS[d])] for d in range(4)])
        trans_node_l = [(0,0)]*4
        trans_node = np.empty(4, dtype=object)
        for r in range(4): trans_node[r] = trans_node_l.copy()
        dead_end = np.zeros((4,4))
        if mirror_direction:
            # store prev_node in each orientation line (transitions) with allowed transition to the access_point
            exit_point = access_point_from_dir(dir)
            targets_found = {agent: (prev_steps - opposite_side_steps, exit_point, target_label) for agent, (opposite_side_steps, target_label) in targets.items()}
            if prev_node is not None:
                for transitions in trans_node[trans[:, exit_point] == 1]: transitions[exit_point] = prev_node
        else:
            # store next_node in each orientation line (transitions) with allowed transition to the exit_point
            exit_point = dir
            targets_found = {agent: (steps, exit_point, target_label) for agent, (steps, target_label) in targets.items()}
            if next_node is not None:
                for transitions in trans_node[trans[:, exit_point] == 1]: transitions[exit_point] = next_node
            if dead_end_detected:
                dead_end[trans[:, exit_point] == 1, exit_point] = (prev_steps*2)
        # if the node already exist proceed with an update
        if (x, y) in self.graph.nodes:
            old_attr = self.graph.nodes[(x, y)]
            for i_o, o in enumerate(trans_node):
                for i_t, t in enumerate(o):
                    if t == (0, 0): trans_node[i_o][i_t] = old_attr["trans_node"][i_o][i_t]
            if len(old_attr["targets"]) != 0:
                targets_found = {**old_attr["targets"], **targets_found}
        return {"trans": trans, "trans_node": trans_node, "dead_end": dead_end, "targets": targets_found}

    def _encode_conflict_node_attributes(self, agent_handle, start, conflict):
        agent = self.env.agents[agent_handle]
        conflict_info = dict()
        conflict_info[DagNodeLabel.CONFLICT] = True
        conflict_info['velocity'] = agent.speed_data['speed']
        conflict_info['conflict_distance'] = nx.shortest_path_length(self.prev_observations[agent_handle], start[0], conflict) + 1
        conflict_info['target_distance'] = start[1]['shortest_path_cost']
        conflict_info['nr_malfunctions'] = agent.malfunction_data['nr_malfunctions']
        conflict_info['next_malfunctions'] = agent.malfunction_data['next_malfunction']
        conflict_info['malfunction_rate'] = agent.malfunction_data['malfunction_rate']
        conflict_info['malfunction'] = agent.malfunction_data['malfunction']
        return {agent_handle: conflict_info}

    def _encode_start_node_attributes(self, agent_handle, switch_distance):
        agent = self.env.agents[agent_handle]
        start_info = dict()
        start_info[DagNodeLabel.START] = True
        start_info['velocity'] = agent.speed_data['speed']
        start_info['switch_distance'] = switch_distance
        start_info['nr_malfunctions'] = agent.malfunction_data['nr_malfunctions']
        start_info['next_malfunctions'] = agent.malfunction_data['next_malfunction']
        start_info['malfunction_rate'] = agent.malfunction_data['malfunction_rate']
        start_info['malfunction'] = agent.malfunction_data['malfunction']
        return start_info

    def _encode_dl_node_attributes(self, dl_steps, first_time_detection):
        return {DagNodeLabel.DEADLOCK: True, "steps_to_deadlock": dl_steps, "first_time_detection": first_time_detection}

    def _get_start_node(self, graph):
        for start_label, start_node in graph.nodes.items():
            if DagNodeLabel.START in start_node: break
        return start_label, start_node

    # Graph utils

    def _is_path_in_graph(self, di_graph, path, source, target):
        undirected_nodes = [(n[0:2]) for n in di_graph.nodes]
        return np.all([node in undirected_nodes for node in path]) and nx.has_path(di_graph, source, target)

    def _copy_graph_structure(self, new_graph, old_graph):
        new_graph.add_nodes_from(old_graph)
        for label, attr in old_graph.nodes.items():
            if DagNodeLabel.START in attr:
                new_graph.update(nodes=[(label,
                         {"shortest_path_cost": attr["shortest_path_cost"], "shortest_path": attr["shortest_path"]})])
            if DagNodeLabel.CONFLICT in attr: continue
            else: new_graph.update(nodes=[(label, attr)])
        for label, attr in old_graph.edges.items():
            new_graph.add_edge(*label, **attr)

    def _print_graph(self, graph, name="graph.png"):
        plt.close()
        nx.draw(graph, with_labels = True, alpha=0.6,
            node_color = 'skyblue', node_size = 1200,
            arrowstyle = '->', arrowsize = 20,
            font_size = 10, font_weight = "bold",
            pos = nx.random_layout(graph))
        plt.savefig(name)

