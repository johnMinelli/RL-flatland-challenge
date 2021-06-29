import networkx as nx
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum, Grid4Transitions
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.observations import TreeObsForRailEnv

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
        self.graph = self._init_graph()
        if self.predictor is not None:
            self.predictor.reset()

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

    def get_next_transition_pos(self, x, y, orientation):
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

    def get_access_directions(self, x, y):
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


    def get(self, handle):

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
                        if self.env.rail.grid[j, i] != 0 and self.is_switch(i, j):
                            # append cell oriented
                            [start_points.append((i, j, new_dir)) for new_dir, accessible in enumerate(self.get_access_directions(i, j)) if accessible]; break
                if len(start_points) != 0: break
            while len(start_points) != 0:
                steps = 0
                targets_in_path = {}
                start_x, start_y, start_dir = start_points.pop()
                x, y, dir = self.get_next_pos(start_x, start_y, start_dir)
                if visited[y, x]: continue
                while True:
                    if self.is_switch(x, y):
                        if start_x == x and start_y == y: break
                        # add switch cell to graph
                        if len(targets_in_path) != 0:
                            old_attr = self.graph.nodes[(start_x, start_y)]
                            new_attr = self.encode_node_attributes(x, y, start_dir, steps, targets_in_path)  # pay attention here
                            new_attr["targets"] = {**old_attr["targets"], **new_attr["targets"]}
                            self.graph.add_node((start_x, start_y), **new_attr)
                        self.graph.add_node((x, y), **self.encode_node_attributes(x, y, self.opposite_dir(dir), steps, targets_in_path))
                        edges.append(((start_x, start_y), (x, y), steps))
                        # and continue visit in other directions
                        [start_points.append((x, y, new_dir)) for new_dir, accessible in enumerate(self.get_access_directions(x, y))
                            if accessible and new_dir != self.opposite_dir(dir)]
                        visited[y, x] = True
                        break
                    elif self.is_dead_end(x, y):
                        break
                    elif visited[y, x]:
                        break
                    else:
                        steps += 1
                        if (x,y) in targets:
                            for agent, target in enumerate(targets):
                                targets_in_path[agent] = steps
                        visited[y, x] = True
                        x, y, dir = self.get_next_transition_pos(x, y, dir)

        self.graph.add_weighted_edges_from(edges)
        print(nx.to_dict_of_dicts(self.graph))
        nx.freeze(self.graph)

    def encode_node_attributes(self, x, y, dir, prev_steps, targets_discovered):
        trans = [self.env.rail.get_transitions(y, x, TRANS[dir]) for dir in range(3)]
        targets = {agent: (prev_steps - opposite_side_steps + 1, dir) for agent, opposite_side_steps in targets_discovered.items()}
        costs = np.zeros((4,4))

        # for col in trans.reverse()[np.max(trans, 0)==1]:
        #     for

        return {"trans": trans, "costs": costs, "targets": targets}

    def opposite_dir(self, dir):
        return (dir+2)%4

    def is_switch(self, x, y):
        all_trans = [self.env.rail.get_transitions(y, x, TRANS[dir]).count(1) for dir in range(4)]
        return np.max(all_trans) > 1

    def is_dead_end(self, x, y):
        return self.env.rail.is_dead_end((y, x))
