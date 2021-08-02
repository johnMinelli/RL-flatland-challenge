import numpy as np
from flatland.core.grid.grid4 import Grid4TransitionsEnum

TRANS = [
    Grid4TransitionsEnum.NORTH,
    Grid4TransitionsEnum.EAST,
    Grid4TransitionsEnum.SOUTH,
    Grid4TransitionsEnum.WEST
]


def get_next_oriented_pos(rail, x, y, orientation, num_step=1):
    """
    get the position after a movement in a direction following transition function
    :param x: current position
    :param y: current position
    :param orientation: current orientation
    :return: the cell position moving forward in facing direction
    """
    bits = format(rail.grid[y, x], 'b').rjust(16, '0')
    for k, bit in enumerate(bits):
        if bit == '1' and int(k / 4) == orientation:
            if k % 4 == 0:
                y -= num_step
            elif k % 4 == 1:
                x += num_step
            elif k % 4 == 2:
                y += num_step
            else:
                x -= num_step
            return x, y, k % 4
    raise Exception('InvalidTransition')

def get_next_pos(x, y, direction, num_step=1):
    """
    get the position after a movement
    :param x: current position
    :param y: current position
    :param direction: direction of movement
    :return: the cell position after the movement
    """
    if direction == 0:
        return x, y - num_step, direction
    elif direction == 1:
        return x + num_step, y, direction
    elif direction == 2:
        return x, y + num_step, direction
    else:
        return x - num_step, y, direction

def get_allowed_directions(rail, x, y):
    """
    get all the positions obtainable after a movement allowed by transition function
    :param x: current position
    :param y: current position
    :return: directions accessible
    """
    access = [False]*4
    bits = format(rail.grid[y, x], 'b').rjust(16, '0')
    for k, bit in enumerate(bits):
        if bit == '1': access[k%4] = True
    return access

def opposite_dir(direction):
    return (direction+2) % 4

def is_switch(rail, x, y, dir=None):
    if dir is None:
        all_trans = [rail.get_transitions(y, x, TRANS[dir]).count(1) for dir in range(4)]
    else:
        all_trans = rail.get_transitions(y, x, TRANS[dir]).count(1)
    return np.max(all_trans) > 1

def is_dead_end(rail, x, y):
    return rail.is_dead_end((y, x))