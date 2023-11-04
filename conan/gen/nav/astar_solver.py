from astar import AStar
import math
from conan.playground import constants
# from core.const import WORLD_WIDTH, WORLD_HEIGHT
import sys

# may be useful for some machines
# sys.setrecursionlimit(5000)

# changed from the demo

def astar_nav(mat_map, start, des):
    Solver = AstarSolver(mat_map)
    way = []
    possible = Solver.astar(tuple(start), tuple(des))
    if possible:
        way = list(possible)
    else:
        way = None
    return way


class AstarSolver(AStar):

    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, mat_map, obj_map=None, target=None):
        self.mat_map = mat_map
        self.obj_map = obj_map
        self.target = target
        self.walkable = []
        # for trees, stones, etc.
        self.doable = []
        for material in constants.walkable:
            self.walkable.append(constants.materials.index(material) + 1)
        # for i in self.walkable:
        #     print(constants.materials[i-1])

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        return 1

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        x, y = node
        neighbors = []
        for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
            flag = True
            if 0 <= nx < 64 and 0 <= ny < 64:
                if self.mat_map[(nx, ny)] not in self.walkable + self.doable:
                    flag = False
                if self.obj_map is not None and self.target is not None:
                    if self.obj_map[(nx, ny)] != 0 and self.obj_map[(nx, ny)] != self.target:
                        flag = False
            else:
                flag = False
            if flag:
                neighbors.append((nx, ny))
        return neighbors
