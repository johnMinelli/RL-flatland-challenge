import networkx as nx
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum, Grid4Transitions
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.observations import TreeObsForRailEnv
import heapq

from src.env.flatland_railenv import FlatlandRailEnv

TRANS = [
    Grid4TransitionsEnum.NORTH,
    Grid4TransitionsEnum.EAST,
    Grid4TransitionsEnum.SOUTH,
    Grid4TransitionsEnum.WEST
]

class ObserverDAG(ObservationBuilder):

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.graph = None

    def set_env(self, env: FlatlandRailEnv):
        super().set_env(env)
        if self.predictor is not None:
            self.predictor.set_env(self.env)

    def reset(self):
        self._init_graph()
        if self.predictor is not None:
            self.predictor.reset(self.graph)

    def get_many(self, handles=None):
        self.predictions = self.predictor.get()

        self.predicted_pos = {}
        for t in range(len(self.predictions[0])):
            pos_list = []
            for a in handles:
                pos_list.append(self.predictions[a][t][1:3])
            # We transform (x,y) coordinates to a single integer number for simpler comparison
            self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
        observations = {}

        # Collect all the different observation for all the agents
        for h in handles:
            observations[h] = self.get(h)
        return observations

    def get(self, handle):
        di_graph = nx.MultiDiGraph() # (node, enter_dir) ----cost, out_dir---->
        working_graph = self.graph.copy()
        edges = []

        start_pos = self.env.agents[handle].initial_position if self.env.agents[handle].position is None else self.env.agents[handle].position
        start_dir = self.env.agents[handle].initial_direction if self.env.agents[handle].direction is None else self.env.agents[handle].direction
        while not self._is_switch(*start_pos):
            if self._is_dead_end(*start_pos): start_dir = self._opposite_dir(start_dir)
            x, y, start_dir = self.get_next_oriented_pos(*start_pos, start_dir) # TODO error
            start_pos = (x, y)
        di_graph.add_node((*start_pos, start_dir))
        # remove unfeasible direction
        feasible_destinations = working_graph.nodes[start_pos]["trans_node"][start_dir]
        for destination, attr in working_graph[start_pos].items():
            if not destination in feasible_destinations:
                working_graph.remove_edge(start_pos, destination)
        # add real target in graph
        target = self.env.agents[handle].target
        di_graph.add_node(target)
        # start exploration and creation of DiGraph
        # add switches for target in DiGraph
        ending_points = []
        for n in self.graph.nodes:
            if handle in n['targets']:
                ending_points.append(n)
                cost, out_dir = n['targets'][handle]
                for access in range(4): # don't you have already it in the node?
                    if self.env.rail.get_transitions(n[1], n[0], TRANS[access])[out_dir] == 1:
                        node = (*n, access)
                        di_graph.add_node(node)
                        edges.append((node, target, {'weight': cost, 'dir': out_dir}))
        #-explore target to agent senza altri agenti
        ''' copia g e togli gli edge di start_pos che non sono in accordo con la direzione
            while dijkstra sul unoriented start to target is in directed
              %la path ovviamente potrebbe essere sbagliata ma la uso per costruire il directed
              itero sulla path aka elenco di nodi:
                 se necessario aggiungo il nodo al directed
                 controllo che la transizione sia valida (considera anche dead-end se la usi costruisci) altrimenti procedo per l'unica alternativa
                 arrivato al nodo guardo se ha tra le info una dead-end, se si e mi permette di continuare la strada iniziale con costo maggiorato
                                  allora l'arco precedente era valido e me lo segno da qualche parte per il futuro costruisco la dead-end nel directed
                                  se invece niente dead-end mi segno l'arco come non valido aka lo rimuovo da una copia del grafo unoriented per il futuro
                                  in entrambi i casi da questo nodo lancio dijkstra e uso questa come nuova path'''
        _, path = nx.multi_source_dijkstra(working_graph, ending_points, start_pos)
        while not self._shortest_path_in_digraph(di_graph, path):
            current_node = None
            current_orientation = start_dir
            i = 0
            path = path[::-1]
            while i < len(path):
                next_node = path[i]
                if current_node is None: current_node = next_node; i+=1; continue
                cost = working_graph[current_node][next_node]["weight"]
                directed_node_destinations = self.graph.nodes[current_node]["trans_node"][current_orientation]
                if current_node in directed_node_destinations and not next_node in directed_node_destinations:
                    # immediate dead_end
                    node_exit_dir = directed_node_destinations.index(current_node)
                    cost = self.graph.nodes[current_node]["dead_end"][current_orientation][node_exit_dir]
                    directed_node_destinations = self.graph.nodes[current_node]["trans_node"][self._opposite_dir(current_orientation)]
                    di_graph.add_node((*current_node, self._opposite_dir(current_orientation))) # TODO remove if no info for node
                    di_graph.add_edge((*current_node, current_orientation), (*current_node, self._opposite_dir(current_orientation)), {'weight': cost})
                    current_orientation = self._opposite_dir(current_orientation)
                if next_node in directed_node_destinations:
                    # transition is valid continue to iterate
                    node_exit_dir = directed_node_destinations.index(next_node)
                    next_orientation = working_graph.edges[(current_node, next_node)]["dir"][next_node]
                    di_graph.add_node((*next_node, next_orientation))
                    di_graph.add_edge((*current_node, current_orientation), (*next_node, next_orientation), {'weight': cost})
                    # update iterators and continue
                    current_node = next_node
                    current_orientation = next_orientation
                    i+=1
                    continue
                else:
                    # this case you must have a single transition possible from your orientation, proceed till a decision and correct the path
                    node_exit_dir = None
                    destination = None
                    for node_exit_dir, destination in enumerate(directed_node_destinations):
                        if destination != (0,0): break
                    if self.graph.nodes[current_node]["dead_end"][current_orientation][node_exit_dir] != 0:
                        # previously detected a dead_end, just compile the DiGraph
                        next_orientation = working_graph.edges[(current_node, destination)]["dir"][destination]
                        total_cost = self.graph.nodes[current_node]["dead_end"][current_orientation][node_exit_dir]
                        partial_cost = working_graph.edges[(current_node, destination)]["weight"]
                        dead_end_cost = ((total_cost - (partial_cost*2))/2)-1
                        self.add_dead_end_path_in_graph(
                            (*current_node, current_orientation), partial_cost, (*destination, next_orientation), dead_end_cost
                            (*destination, self._opposite_dir(next_orientation)), (*current_node, self._opposite_dir(current_orientation))
                        )
                        # update iterators and continue
                        current_orientation = self._opposite_dir(current_orientation)
                        continue
                    # check dead_end presence
                    unreachable_node = next_node
                    next_node = destination
                    next_orientation = working_graph.edges[(current_node, next_node)]["dir"][next_node]
                    cost = working_graph.edeges[(current_node, next_node)]["weight"]
                    next_node_destinations = self.graph.nodes[next_node]["trans_node"][next_orientation]
                    if next_node in next_node_destinations:
                        # dead_end found
                        next_node_exit_dir = directed_node_destinations.index(next_node)
                        partial_cost = cost
                        dead_end_cost = self.graph.nodes[next_node]["dead_end"][next_orientation][next_node_exit_dir]
                        self.graph.nodes[current_node]["dead_end"][current_orientation][node_exit_dir] = (partial_cost*2)+dead_end_cost*2-1
                        self.add_dead_end_path_in_graph(
                            (*current_node, current_orientation), partial_cost, (*next_node, next_orientation), dead_end_cost
                            (*next_node, self._opposite_dir(next_orientation)), (*current_node, self._opposite_dir(current_orientation))
                        )
                    else:
                        # the transition for this direction is invalid
                        working_graph.remove_edge((current_node, unreachable_node))
                    # re launch dijkstra
                    _, path = nx.multi_source_dijkstra(working_graph, ending_points, next_node)
                    path = path[::-1]
                    current_node = None
                    current_orientation = next_orientation
                    i = 0
        #-explore target to agent con altri agenti
            # TODO
        #explore each node of (di or not di?)graph for paths to loose time
            # TODO

        observation = np.zeros(10)

        # We track what cells where considered while building the observation and make them accessible for rendering
        visited = set()

        for _idx in range(10):
            # Check if any of the other prediction overlap with agents own predictions
            x_coord = self.predictions[handle][_idx][1]
            y_coord = self.predictions[handle][_idx][2]

            # We add every observed cell to the observation rendering
            visited.add((x_coord, y_coord))
            if self.predicted_pos[_idx][handle] in np.delete(self.predicted_pos[_idx], handle, 0):
                # We detect if another agent is predicting to pass through the same cell at the same predicted time
                observation[handle] = 1

        # This variable will be accessed by the renderer to visualize the observations
        self.env.dev_obs_dict[handle] = visited

        return observation

    def _init_graph(self):
        self.graph = nx.Graph()
        # create an edge for each pair of connected switch
        visited = np.zeros((self.env.width, self.env.height), np.bool)
        targets = [a.target for a in self.env.agents]
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
        nx.freeze(self.graph)

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

    def encode_node_attributes(self, x, y, dir, prev_steps, targets, next_node=None, prev_node=None, mirror_direction=False, dead_end_detected=False):
        trans = np.array([[*self.env.rail.get_transitions(y, x, TRANS[dir])] for dir in range(4)])
        trans_node = np.array(([[(0,0)]*4])*4) # TODO error no tuple
        dead_end = np.zeros((4,4))
        costs = np.zeros((4,4))
        if mirror_direction:
            # per ogni posizione con sbocco in _opposite_dir(dir) ha prev_node
            dir = self._opposite_dir(dir)
            targets = {agent: (prev_steps - opposite_side_steps + 1, dir) for agent, opposite_side_steps in targets.items()}
            if not prev_node is None:
                trans_node[trans[:, dir] == 1, dir] = prev_node
        else:
            if not next_node is None:
                trans_node[trans[:, dir] == 1, dir] = next_node
        if dead_end_detected:
            dead_end[trans[:, dir] == 1, dir] = (prev_steps*2)-1
        # if the node already exist is an update
        if (x,y) in self.graph.nodes:
            old_attr = self.graph.nodes[(x,y)]
            trans_node = np.where(trans_node == 0, old_attr["trans_node"], trans_node)
            if len(targets) != 0:
                targets = {**old_attr["targets"], **targets}
        return {"trans": trans, "trans_node": trans_node, "costs": costs, "targets": targets}

    def _opposite_dir(self, direction):
        return (direction+2) % 4

    def _is_switch(self, x, y):
        all_trans = [self.env.rail.get_transitions(y, x, TRANS[dir]).count(1) for dir in range(4)]
        return np.max(all_trans) > 1

    def _is_dead_end(self, x, y):
        return self.env.rail.is_dead_end((y, x))

    def _shortest_path_in_digraph(self, di_graph, path):
        undirected_nodes = [(n[0:1]) for n in di_graph.nodes]
        return np.all([node in undirected_nodes for node in path])

    def add_dead_end_path_in_graph(self,di_graph, a_front, ab_cost, b_front, bb_cost, b_back, a_back):
        di_graph.add_node(b_front)
        di_graph.add_edge(a_front, b_front, {'weight': ab_cost})
        # get back through the dead_end
        di_graph.add_node(b_back)
        di_graph.add_edge(b_front, b_back, {'weight': bb_cost})
        di_graph.add_node(a_back)
        di_graph.add_edge(b_back, a_back, {'weight': ab_cost})