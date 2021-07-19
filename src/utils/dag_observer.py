import networkx as nx
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum, Grid4Transitions
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.observations import TreeObsForRailEnv
import heapq

from src.env.flatland_railenv import FlatlandRailEnv

# handle get many to call get; add priority index to call in that order low to high;
# in get compute observation graph only on need otherwise reuse the previous structure but recompute deadlock info each time
# handle deadlock in pre start switch as two nodes graph
# start, target, conflict, starvation, deadlock, deadend
# nodo causa deadlock informazioni dell'agente adversary
'dist_own_target_encountered '
'dist_position_to_conflict '
'dist_min_to_target '
'num_agents_malfunctioning '
'speed_min_fractional '

# reason about hen you are in fron of the target    V
# reason about presence of an agent on switch       V
# reason about deadlock on switch                   V
# reason about agent without path to target         V

# when no path to target is admissible:
# primary choice is dead_edge
# secondary choice is dead_end
# remember that the deadlock label will be applied only when two agents TOUCH into an edge


TRANS = [
    Grid4TransitionsEnum.NORTH,
    Grid4TransitionsEnum.EAST,
    Grid4TransitionsEnum.SOUTH,
    Grid4TransitionsEnum.WEST
]


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
        di_graph = nx.MultiDiGraph()  # (node, enter_dir) ----cost, out_dir---->
        working_graph = self.graph

        other_agents_position = [(*(a.initial_position if a.position is None else a.position), a.initial_direction if a.position is None else a.direction)
                                 for iter_handle, a in enumerate(self.env.agents) if iter_handle != handle]
        steps = 0
        target_flag = False
        deadlock_flag = False

        #prima di tutto check dal controller che l'agent non sia in deadlock else return NONE (that shuoldn't happen anymore)
        #TODO quando c'è deadlock ritorna labeled node con passi predeadlock come costo (quando passi=0 allora l'agente è vero deadlock)
        #quando hai deadlock a 0 (ignorando le caselle com agenti nella tua stessa direzione) step davanti a te e 0 dallo switch dietro a te remove edge one direction
        start_pos = (self.env.agents[handle].initial_position if self.env.agents[handle].position is None else self.env.agents[handle].position)[::-1]
        start_dir = self.env.agents[handle].initial_direction if self.env.agents[handle].direction is None else self.env.agents[handle].direction
        target = self.env.agents[handle].target[::-1]
        while not self._is_switch(*start_pos, start_dir):
            if start_pos == target: target_flag = True; break
            #if (*start_pos, self._opposite_dir(start_dir)) in other_agents_position: deadlock_flag = True
            if self._is_dead_end(*start_pos): start_dir = self._opposite_dir(start_dir)
            x, y, start_dir = self.get_next_oriented_pos(*start_pos, start_dir)
            start_pos = (x, y)
            steps += 1
        di_graph.add_node((*start_pos, start_dir), **{"start": True, **self.encode_start_node_attributes(handle, steps)})
        if target_flag:
            di_graph.update(nodes=[((*start_pos, start_dir), {"target": True})])
            return di_graph
        if self.env.dl_controller.deadlocks[handle]:
            di_graph.update(nodes=[((*start_pos, start_dir), {"deadlock": True})])
            return di_graph

        if steps != 1 and not len(self.prev_observations) != 0 and handle in self.prev_observations: #TODO return None for not decision required
            # reuse previous observation structure
            di_graph = self._copy_graph_structure(self.prev_observations[handle])
        else:
            # remove edges with a deadlock
            for dead_edge in self.env.dl_controller.negated_edges:
                self._remove_edge_and_transition(working_graph, *dead_edge, both_directions=True)
            self.env.dl_controller.negated_edges = dict()

            # add real target in graph
            di_graph.add_node(target, **{"target": True})
            # start exploration and building of DiGraph
            # add switches near agent's target in DiGraph
            ending_points = []
            for _, n in working_graph.nodes.items():
                if handle in n['targets']:
                    ending_points.append(n)
                    cost, out_dir = n['targets'][handle]
                    for access in range(4):
                        if n["trans"][access][out_dir] == 1:
                            node = (*n, access)
                            di_graph.add_node(node)
                            di_graph.add_edge(node, target, {'weight': cost, 'dir': out_dir})

            # -explore target-to-agent without other agents
            if not self._build_paths_in_directed_graph(working_graph.copy(True), di_graph, start_pos, start_dir, ending_points):
                di_graph.update(nodes=[((*start_pos, start_dir), {"starvation": True})])
                #TODO act of consequence and send the agent in some non disturbing place
            # -explore path target-to-agent other agents
            for a in np.array(self.env.agents)[not self.env.dl_controller.deadlocks]:
                if a.status == RailAgentStatus.ACTIVE or a.status == RailAgentStatus.READY_TO_DEPART:
                    # remove busy edges (only opposite direction) from working_graph
                    pos_y, pos_x = a.initial_position if a.position is None else a.position
                    dir = a.initial_direction if a.position is None else a.direction
                    if self._is_switch(pos_x, pos_y, dir):
                        for iter_dir in range(4):
                            t = working_graph.nodes[(pos_x, pos_y)]["trans_node"][:,iter_dir]
                            [self._remove_edge_and_transition(working_graph, (pos_x, pos_y), destination_node) for destination_node in t if destination_node != (0,0)]
                    else:
                        while True:
                            if self._is_dead_end(pos_x, pos_y): dir = self._opposite_dir(dir)
                            if self._is_switch(pos_x, pos_y, dir): break
                            pos_x, pos_y, dir = self.get_next_oriented_pos(pos_x, pos_y, dir)
                            steps += 1
                        t = working_graph.nodes[(pos_x, pos_y)]["trans_node"][:, self._opposite_dir(dir)]
                        [self._remove_edge_and_transition(working_graph, (pos_x, pos_y), destination_node) for destination_node in t if destination_node != (0,0)]
            self._build_paths_in_directed_graph(working_graph.copy(True), di_graph, start_pos, start_dir, ending_points)

        # add attributes to nodes based on conflicts
        agent_ranking_pos = self.prioritized_agents.index(handle)
        for matching_handle in range(agent_ranking_pos):
            # a conflict node is a common node between the obs DiGraph and the other agent DiGraph (compare only x, y)
            if not self.prev_observations[matching_handle] is None:
                matching_graph = self.prev_observations[matching_handle]
                start = None
                for _, start in matching_graph.nodes.items():
                    if "start" in start: break
                node_list = set()
                [node_list.update(nx.descendants_at_distance(matching_graph, start, radius)) for radius in range(self.conflict_radius)]
                possible_conflicts = set()
                [possible_conflicts.update([(*matching_node[0:2], other_dir) for other_dir in range(4) if other_dir != matching_node[2]]) for matching_node in node_list]
                for conflict_node in possible_conflicts.intersection(di_graph.nodes.keys()):
                    matched_conflict_node = set([(*conflict_node[0:2],d) for d in range(4)]).intersection(node_list)[0]
                    di_graph.update(nodes=[(conflict_node, {"conflict": True, **self.encode_conflict_node_attributes(matching_handle, start, matched_conflict_node)})])

        observation = np.zeros(10)

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
        self.graph = nx.Graph()
        # create an edge for each pair of connected switch
        visited = np.zeros((self.env.width, self.env.height), np.bool)
        targets = [a.target[::-1] for a in self.env.agents]
        edges = []
        start_points = []
        while not np.all(visited[self.env.rail.grid>0]):
            for i, row in enumerate(self.env.rail.grid):
                for j, _ in enumerate(row):
                    if not visited[j, i] and (i,j) not in targets:
                        if self.env.rail.grid[j, i] != 0 and self._is_switch(i, j):
                            # append cell oriented
                            [start_points.append((i, j, new_dir)) for new_dir, accessible in enumerate(self.get_allowed_directions(i, j)) if accessible]; break
                if len(start_points) != 0: break
            while len(start_points) != 0:
                steps = 0
                targets_in_path = {}
                start_x, start_y, start_dir = start_points.pop()
                x, y, dir = self.get_next_pos(start_x, start_y, start_dir)
                edge_entering_dir = self._opposite_dir(dir)
                if visited[y, x]: continue
                while True:
                    if self._is_switch(x, y):
                        if start_x == x and start_y == y: break
                        # update previous node transition
                        self.graph.add_node((start_x, start_y), **self.encode_node_attributes(start_x, start_y, start_dir, steps, targets_in_path, next_node=(x, y)))
                        # add current switch cell to graph with mirrored information
                        self.graph.add_node((x, y), **self.encode_node_attributes(x, y, dir, steps, targets_in_path, prev_node=(start_x, start_y), mirror_direction=True))
                        edges.append(((start_x, start_y), (x, y), {'weight': steps, 'dir': {(start_x, start_y):edge_entering_dir, (x, y): dir}})) # entering_direction
                        # and continue visit in other directions
                        [start_points.append((x, y, new_dir)) for new_dir, accessible in enumerate(self.get_allowed_directions(x, y))
                         if accessible and new_dir != self._opposite_dir(dir)]
                        visited[y, x] = True
                        break
                    elif self._is_dead_end(x, y):
                        self.graph.add_node((start_x, start_y), **self.encode_node_attributes(start_x, start_y, start_dir, steps, targets_in_path, dead_end_detected=True))
                    elif visited[y, x]:
                        break
                    else:
                        steps += 1
                        if (x,y) in targets:
                            for agent, target in enumerate(targets):
                                targets_in_path[agent] = steps
                        visited[y, x] = True
                        x, y, dir = self.get_next_oriented_pos(x, y, dir)

        self.graph.add_weighted_edges_from(edges)
        print(nx.to_dict_of_dicts(self.graph))
        self.graph = self.graph.to_directed()
        nx.freeze(self.graph)

    def _build_paths_in_directed_graph(self, exploration_graph, directed_graph, start_pos, start_dir, ending_points):
        path = self._get_shorthest_path(exploration_graph, ending_points, start_pos, start_dir) # TODO shortest_path_cost for start_node
        if path is None: return False
        while not self._is_path_in_graph(directed_graph, path):
            current_node = None
            current_orientation = start_dir
            i = 0
            path = path[::-1]
            while i < len(path):
                next_node = path[i]
                if current_node is None: current_node = next_node; i += 1; continue
                node_directed_destinations = exploration_graph[current_node]["trans_node"][current_orientation]

                if next_node in node_directed_destinations:
                    # transition is valid continue to iterate
                    node_exit_dir = node_directed_destinations.index(next_node)
                    next_orientation = exploration_graph.edges[(current_node, next_node)]["dir"][next_node]
                    cost, destination, dead_end_flag = self._update_graph_until_switch(exploration_graph,
                                                                                       current_node, current_orientation,
                                                                                       next_node, next_orientation)
                    directed_graph.add_node((*next_node, next_orientation))
                    directed_graph.add_edge((*current_node, current_orientation), destination,{'weight': cost, "out_dir": node_exit_dir})
                    # update iterators and continue
                    current_node = destination[0:2]
                    current_orientation = destination[2]
                    i += 1
                    continue
                elif current_node in node_directed_destinations:
                    # immediate dead_end from current_node
                    node_exit_dir = node_directed_destinations.index(current_node)
                    cost, destination, dead_end_flag  = self._update_graph_until_switch(exploration_graph,
                                                                                        current_node, current_orientation,
                                                                                        current_node, self._opposite_dir(current_orientation))
                    directed_graph.add_node(destination, **{"dead_end": True})
                    directed_graph.add_edge((*current_node, current_orientation), destination,
                                            {'weight': cost, "out_dir": node_exit_dir})
                    # update iterators and continue flow
                    current_node = destination[0:2]
                    current_orientation = destination[2]
                    continue
                else:
                    # the transition for this direction is invalid
                    self._remove_edge_and_transition(exploration_graph, current_node, next_node)
                    # this case you must have at least a transition possible from your orientation
                    unreachable_node = next_node
                    exploration_node = None
                    next_node = None
                    for node_exit_dir, destination_node in enumerate(node_directed_destinations):
                        if destination_node == (0, 0): continue
                        
                        if unreachable_node in exploration_graph.nodes[current_node]["trans_node"][self._opposite_dir(node_exit_dir)]:
                            # transition which eventually with a dead and can take you to the unreachable_node
                            next_orientation = exploration_graph.edges[(current_node, destination_node)]["dir"][destination_node]
                            cost, destination, dead_end_flag  = self._update_graph_until_switch(exploration_graph,
                                                                                                current_node, current_orientation,
                                                                                                destination_node, next_orientation)
                            destination_node = destination[0:2]
                            next_orientation = destination[2]
                            next_node_directed_destinations = exploration_graph.nodes[destination_node]["trans_node"][next_orientation]
                            if destination_node in next_node_directed_destinations:
                                # (path resolution) immediate dead_end
                                directed_graph.add_edge((*current_node, current_orientation),(*destination_node, next_orientation),
                                                        {'weight': cost, "out_dir": node_exit_dir})
                                dead_end_cost, dead_end_destination, _ = self._update_graph_until_switch(exploration_graph,
                                                                                                      destination_node, next_orientation,
                                                                                                      destination_node, self._opposite_dir(next_orientation))
                                directed_graph.add_node(dead_end_destination, **{"dead_end": True})
                                directed_graph.add_edge((*destination_node, next_orientation), dead_end_destination,
                                                        {'weight': dead_end_cost, "out_dir": node_directed_destinations.index(destination_node)})
                            # else:
                                # (path resolution) 1 step ahead dead_end
                        elif exploration_node is None: exploration_node = destination_node
                        if next_node is None: next_node = destination_node

                    # need a new dijkstra iteration
                    next_node = next_node if exploration_node is None else exploration_node
                    next_orientation = exploration_graph.edges[(current_node, next_node)]["dir"][next_node]
                    cost = exploration_graph.edges[(current_node, next_node)]["weight"]
                    node_exit_dir = node_directed_destinations.index(next_node)
                    directed_graph.add_edge((*current_node, current_orientation), (*next_node, next_orientation),
                                            {'weight': cost, "out_dir": node_exit_dir})
                    path = self._get_shorthest_path(exploration_graph, ending_points, next_node, node_exit_dir)
                    if path is None: return True
                    current_node = None
                    current_orientation = next_orientation
                    i = 0
                    path = path[::-1]
            path = self._get_shorthest_path(exploration_graph, ending_points, next_node, node_exit_dir)
            if path is None: return True

    def _is_path_in_graph(self, di_graph, path):
        undirected_nodes = [(n[0:1]) for n in di_graph.nodes]
        return np.all([node in undirected_nodes for node in path])
        nx.find_cycle()

    def _update_graph_until_switch(self, undirected_graph, current_node, current_orientation, next_node, next_orientation):
        # find switch
        # update undirected graph with new edges end remove the one not relevant (no switch)
        # update transition matrix of current_node in row current_orientation with real node switch
        start_node_exit_dir = undirected_graph[current_node]["trans_node"][current_orientation].index(next_node)
        if current_node == next_node:
            cost = self.graph.nodes[current_node]["dead_end"][current_orientation][start_node_exit_dir] + 1
        else:
            cost = undirected_graph.edges[(current_node, next_node)]["weight"] + 1
        undirected_graph.remove_edge(current_node, next_node)

        node_exit_dir = None
        dead_end = False
        while not self._is_switch(*next_node, next_orientation):
            next_node_directed_destinations = undirected_graph[next_node]["trans_node"][next_orientation]
            for node_exit_dir, destination_node in enumerate(next_node_directed_destinations):
                if destination_node != (0, 0):
                    # remove original edges but leave untouched the transitions
                    undirected_graph.remove_edge(next_node, destination_node)
                    if next_node == destination_node:
                        dead_end = True
                        next_orientation = self._opposite_dir(next_orientation)
                        cost += undirected_graph.nodes[current_node]["dead_end"][current_orientation][node_exit_dir] + 1
                    else:
                        next_orientation = undirected_graph.edges[(next_node, destination_node)]["dir"][destination_node]
                        cost += undirected_graph.edges[(current_node, next_node)]["weight"] + 1
                    next_node = destination_node
                    break
            if current_node == next_node:
                break
        undirected_graph.add_edge((*current_node, current_orientation), (*next_node, next_orientation), {'weight': cost, "out_dir": node_exit_dir})
        undirected_graph[current_node]["trans_node"][current_orientation][start_node_exit_dir] = next_node
        return cost, (*next_node, next_orientation), dead_end

    def _remove_edge_and_transition(self, graph, node1, node2, both_directions=False):
        graph.remove_edge(node1, node2)
        transitions_node = graph.nodes[node1]["trans_node"]
        transitions = graph.nodes[node1]["trans"]
        for row, destinations in enumerate(transitions_node):
            for col, destination_node in enumerate(destinations):
                if destination_node == node2:
                    transitions_node[row][col] = (0,0)
                    transitions[row][col] = 0
        graph.nodes[node1]["trans_node"] = transitions_node
        graph.nodes[node1]["trans"] = transitions
        if both_directions: self._remove_edge_and_transition(graph, node2, node1)

    def _get_shorthest_path(self, graph, sources, target, allowed_target_dir):
        reduced_graph = graph.copy()
        # remove unfeasible directions
        feasible_destinations = reduced_graph.nodes[target]["trans_node"][allowed_target_dir]
        for destination, attr in reduced_graph[target].items():
            if not destination in feasible_destinations:
                reduced_graph.remove_edge(destination, target)
        try:
            _, path = nx.multi_source_dijkstra(graph, sources, target)
            return path
        except:
            return None

    def _copy_graph_structure(self, param):
        pass

    def _rank_agents(self):
        list = dict()
        if len(self.prev_observations) == 0:
            return [a.handle for a in
                    sorted(self.env.agents, key=lambda agent: agent.speed_data["speed"], reverse=True)]
        else:
            for handle, agent in enumerate(self.env.agents):
                start_node = None
                for _, start_node in self.prev_observations[handle].nodes.items():
                    if "start" in start_node: break;
                next_malfunction = agent.malfunction_data["next_malfunction"]
                malfunction = agent.malfunction_data["malfunction"]
                velocity = agent.speed_data["speed"]
                switch = start_node["shortest_path"]
                distance = start_node["shortest_path_cost"]
                ratio = ((distance / (next_malfunction + 1 / (malfunction + 1))) / velocity) * switch
                list.update({handle: ratio})
        return dict(sorted(list.items(), key=lambda x: x[1])).keys()

    def get_next_oriented_pos(self, x, y, orientation):
        """
        get the position after a movement in a direction following transition function
        :param x: current position
        :param y: current position
        :param orientation: current orientation
        :return: the cell position moving forward in facing direction
        """
        bits = format(self.env.rail.grid[y, x], 'b').rjust(16, '0')
        for k, bit in enumerate(bits):
            if bit == '1' and int(k / 4) == orientation:
                if k % 4 == 0:
                    y -= 1
                elif k % 4 == 1:
                    x += 1
                elif k % 4 == 2:
                    y += 1
                else:
                    x -= 1
                return x, y, k % 4
        raise Exception('InvalidTransition')

    def get_next_pos(self, x, y, direction):
        """
        get the position after a movement
        :param x: current position
        :param y: current position
        :param direction: direction of movement
        :return: the cell position after the movement
        """
        if direction == 0:
            return x, y - 1, direction
        elif direction == 1:
            return x + 1, y, direction
        elif direction == 2:
            return x, y + 1, direction
        else:
            return x - 1, y, direction

    def get_allowed_directions(self, x, y):
        """
        get all the positions obtainable after a movement allowed by transition function
        :param x: current position
        :param y: current position
        :return: directions accessible
        """
        access = [False]*4
        bits = format(self.env.rail.grid[y, x], 'b').rjust(16, '0')
        for k, bit in enumerate(bits):
            if bit == '1': access[k%4] = True
        return access

    def _opposite_dir(self, direction):
        return (direction+2) % 4

    def _is_switch(self, x, y, dir=None):
        if dir is None:
            all_trans = [self.env.rail.get_transitions(y, x, TRANS[dir]).count(1) for dir in range(4)]
        else:
            all_trans = self.env.rail.get_transitions(y, x, TRANS[dir]).count(1)
        return np.max(all_trans) > 1

    def _is_dead_end(self, x, y):
        return self.env.rail.is_dead_end((y, x))

    def encode_node_attributes(self, x, y, dir, prev_steps, targets, next_node=None, prev_node=None, mirror_direction=False, dead_end_detected=False):
        trans = np.array([[*self.env.rail.get_transitions(y, x, TRANS[dir])] for dir in range(4)])
        trans_node_l = ([[(0,0)]*4])*4
        trans_node = np.empty(len(trans_node_l), dtype=object)
        trans_node[:] = trans_node_l
        dead_end = np.zeros((4,4))
        if mirror_direction:
            # per ogni posizione con sbocco in _opposite_dir(dir) ha prev_node
            dir = self._opposite_dir(dir)
            targets = {agent: (prev_steps - opposite_side_steps + 1, dir) for agent, opposite_side_steps in targets.items()}
            if not prev_node is None:
                for transitions in trans_node[trans[:, dir] == 1]: transitions[dir] = prev_node
        else:
            if not next_node is None:
                for transitions in trans_node[trans[:, dir] == 1]: transitions[dir] = next_node
        if dead_end_detected:
            dead_end[trans[:, dir] == 1, dir] = (prev_steps*2)-1
        # if the node already exist is an update
        if (x,y) in self.graph.nodes:
            old_attr = self.graph.nodes[(x,y)]
            for arry,o in enumerate(trans_node):
                for arrx,t in enumerate(o):
                    if t == (0, 0): trans_node[arry][arrx] = old_attr["trans_node"][arry][arrx]
            if len(old_attr["targets"]) != 0:
                targets = {**old_attr["targets"], **targets}
        return {"trans": trans, "trans_node": trans_node, "dead_end": dead_end, "targets": targets}

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
        start_info['next_malfunctions'] = agent.malfunction_data['next_malfunctions']
        start_info['malfunction_rate'] = agent.malfunction_data['malfunction_rate']
        start_info['malfunction'] = agent.malfunction_data['malfunction']
        return start_info