from conan.gen.nav.astar_solver import AstarSolver
from random import random, randint
import conan.playground.constants as constants
import conan.gen.task.task_constants as task_constants
from conan.gen.task.check import valid_check
import numpy as np
import random
from conan.gen.task.parser import parse_task


class TaskAgent:

    def __init__(self, task_type, task, pos, mat_map, obj_map, inventory=None):
        self._mat_map = mat_map
        self._obj_map = obj_map
        self._task_type = task_type
        self._pos = tuple(pos)
        self._task = task
        self._curr_task = None
        self._sub_tasks = []
        self._path = []
        # stack for subtasks info
        self._q_stack = []
        # list of dict {task, start, des} to save finished task information
        self._task_log = []
        self._facing = (1, 0)
        self._last_finished = True
        self._inventory = inventory
        self._mat_ids = {x: i for i, x in enumerate(
            [None] + constants.materials)}
        self._obj_ids = {"cow": 2, "zombie": 3, "skeleton": 4}

    def parse_task(self):
        self._sub_tasks = parse_task(self._task_type, self._task)

    def get_path(self, des=None, target_id=None, doable=[]):
        # print(f"--- Astar, des {des}, now {self._pos}, path {self._path} ---")
        self._path = []
        solver = AstarSolver(self._mat_map, self._obj_map, target_id)
        solver.doable = doable
        possible = solver.astar(tuple(self._pos), tuple(des))
        if possible:
            self._path = list(possible)[1:]
            # print(f"--- Astar, des {des}, now {self._pos}, path {self._path} ---")
            return True
        else:
            if self._inventory != None and self._inventory["wood_pickaxe"] > 0:
                # to be more natural, set doable at second trial here
                solver.doable = [3, 6] + doable
                # print("second trial")
                possible = solver.astar(tuple(self._pos), tuple(des))
                if possible:
                    self._path = list(possible)[1:]
                    return True
            return False

    def move(self, doable=[], target_id=None, object=False):
        action = None
        done = False
        map = self._mat_map if not object else self._obj_map
        directions = dict(left=(-1, 0), right=(+1, 0),
                          up=(0, -1), down=(0, +1))
        if len(self._path) == 0:
            return True, 'noop', self._pos
        next_step = self._path[0]
        if not valid_check(next_step):
            return False, 'noop', self._pos
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
        facing_pos = tuple(np.array(self._pos) + np.array(self._facing))
        self._facing = directions[action.split('_')[1]]
        if map[next_step] == target_id and next_step == facing_pos:
            action = 'noop'
            done = True
            next_step = self._pos
        elif self._mat_map[next_step] in doable and next_step != facing_pos:
            next_step = self._pos
            done = True
        else:
            self._pos = next_step
            self._path.pop(0)
        return done, action, next_step

    def find(self, target):
        if target in self._obj_ids.keys():
            target = self._obj_ids[target]
            targets = np.where(self._obj_map == target)
        else:
            target = self._mat_ids[target]
            targets = np.where(self._mat_map == target)
        targets = np.array(list(zip(targets[0], targets[1])))
        # find the nearest one with self._pos
        if len(targets) == 0:
            return None
        idx = np.abs((targets - self._pos)).sum(axis=1).argmin(axis=0)
        return targets[idx]

    def goto(self, target):
        # find, get path, move
        # target = "diamond"
        des = self.find(target)
        if des is None:
            return "reset", self._pos
        target_id = self._mat_ids[target]
        self.get_path(des, target_id, doable=[target_id])
        if len(self._path):  # find path
            self._last_finished = False
            done, action, target = self.move(
                doable=[target_id], target_id=target_id)
            self._last_finished = done
            return action, target
        else:
            return "reset", self._pos

    def do(self, target):
        if target in self._obj_ids.keys():
            target = self._obj_ids[target]
        else:
            target = self._mat_ids[target]
        facing_pos = tuple(np.array(self._pos) + np.array(self._facing))
        if self._last_finished and self._mat_map[facing_pos] != target:
            # print("Not at the target!")
            return "noop", self._pos
        self._last_finished = False
        action = 'do'
        if self._mat_map[facing_pos] != target:
            self._last_finished = True
            return "noop", self._pos
        self._last_finished = True
        return action, self._pos

    def place(self, target):
        available_places = constants.place[target]["where"]
        available_places = [self._mat_ids[x] for x in available_places]
        action = 'place_' + target
        while len(self._path) <= 1:
            des = self._pos + np.array([randint(-9, 9), randint(-9, 9)])
            self.get_path(des)

        if target in self._obj_ids.keys():
            target = self._obj_ids[target]
        else:
            target = self._mat_ids[target]
        facing_pos = tuple(np.array(self._pos) + np.array(self._facing))
        self._last_finished = False
        if valid_check(facing_pos) and self._mat_map[facing_pos] in available_places:
            self._last_finished = True
            self._path = []
            return action, self._pos
        else:
            _, action, target = self.move()
            return action, target

    def make(self, target):
        return "make_" + target, self._pos

    def chase(self, target):
        # print("Chasing", target)
        des = self.find(target)
        if des is None:
            return "reset", self._pos
        target_id = self._obj_ids[target]
        # indicate for the dead target
        dead_id = self._mat_ids[target + "_dead"]
        self._last_finished = False
        facing_pos = tuple(np.array(self._pos) + np.array(self._facing))
        if valid_check(facing_pos) and self._obj_map[facing_pos] == target_id:
            return "do", self._pos
        elif valid_check(facing_pos) and self._mat_map[facing_pos] == dead_id:
            self._last_finished = True
            return "noop", self._pos
        if not self._path or len(self._path) < 5:
            self.get_path(des, doable=[target_id])
        _, action, next_step = self.move(
            doable=[], target_id=target_id, object=True)
        return action, next_step

    def action(self):
        if self._last_finished:
            if len(self._sub_tasks) == 0:
                return "noop", self._pos
            self._curr_task = self._sub_tasks.pop(0)
            while self._curr_task[0] in ["[", "]"]:
                self.set_stack(self._curr_task)
                if len(self._sub_tasks) == 0:
                    return "noop", self._pos
                self._curr_task = self._sub_tasks.pop(0)
        action, target = self._curr_task.split('_', 1)
        func = getattr(self, action)
        return func(target)

    def set_stack(self, task):
        if task[0] == "[":
            task_dict = dict()
            task_dict["task"] = task[1:]
            task_dict["start"] = str(self._pos)
            task_dict["des"] = None
            self._q_stack.append(task_dict)
        elif task == "]":
            task_dict = self._q_stack.pop()
            task_dict["des"] = str(self._pos)
            self._task_log.append(task_dict)
            # print(task_dict)
            # print(self._task_log)

    def update(self, mat_map, obj_map, agent_inventory):
        self._mat_map = mat_map
        self._obj_map = obj_map
        self._inventory = agent_inventory
