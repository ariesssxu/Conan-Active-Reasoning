from conan.gen.nav.astar_solver import AstarSolver
from random import random, randint
import numpy as np


class AstarAgent:

    def __init__(self, start, des, mat_map):
        self._start = tuple(start)
        self._pos = tuple(start)
        self._des = tuple(des)
        self._mat_map = mat_map
        self._path = []
        self._solver = AstarSolver(mat_map)
        # print(f"--- Start from {start}, Destination {des}, now {self.pos} ---")

    def get_path(self, des=None):
        # print(f"--- Astar, des {self.des}, now {self.pos} ---")
        possible = self._solver.astar(tuple(self._pos), tuple(self._des)) if des is None else \
            self._solver.astar(tuple(self._pos), tuple(des))
        if possible:
            self._path = list(possible)[1:]
            return True
        else:
            return False

    def move(self):
        action = None
        next_step = self._path[0]
        if next_step[1] == self._pos[1]:
            if next_step[0] > self._pos[0]:
                action = 'move_right'
            else:
                action = 'move_left'
        else:
            if next_step[1] > self._pos[1]:
                action = 'move_down'
            else:
                action = 'move_up'
        self._pos = next_step
        self._path.pop(0)
        return action, next_step

    def action(self):
        # print(f"--- Start from {self.start}, Destination {self.des}, now {self.pos} ---")
        if self._pos == self._des:
            return "sleep", self._des
        if len(self._path) == 0:
            return None
        else:
            return self.move()

    def update(self, mat_map, obj_map):
        self._mat_map = mat_map
        self._obj_map = obj_map


class NoisyAstarAgent(AstarAgent):

    def __init__(self, start, des, mat_map):
        super().__init__(start, des, mat_map)
        self.noise = 0.2
        self.noise_count = 0
        self.noise_limit = 5
        self.ori_des = None
        self.noise_flag = False

    def add_noise(self):
        bias = [randint(-10, 10), randint(-10, 10)]
        if self.get_path(np.array(self._pos) + bias):
            self._ori_des = self._des
            self._des = tuple(np.array(self._pos) + bias)
            self.noise_count += 1
            self.noise_flag = True
            print("Noisy path added.")
        else:
            return
            print("Add noisy path failed.")

    def action(self):
        # print(f"--- Start from {self.start}, Destination {self.des}, now {self.pos} ---")
        if random() < self.noise and self.noise_count < self.noise_limit and not self.noise_flag:
            self.add_noise()
        if self.noise_flag and self._pos == self._des:
            self.noise_flag = False
            self._des = self._ori_des
            self._ori_des = None
            self.get_path()
            print("Back to origin path.")
        if self._pos == self._des:
            return "sleep", self._des
        if len(self._path) == 0:
            return None
        else:
            return self.move()

    def update(self, mat_map, obj_map):
        self._mat_map = mat_map
        self._obj_map = obj_map
