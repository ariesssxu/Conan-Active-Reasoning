from conan.gen.nav.astar_solver import AstarSolver
import random
import conan.gen.survival.survival_constants as survival_constants
import numpy as np
from conan.gen.task.parser import parse_task
from conan.gen.task.agents import TaskAgent
import sys

# may be useful for some machines
# sys.setrecursionlimit(5000)

def isfar(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) > 10


class SurvivalAgent(TaskAgent):

    def __init__(self, pos, mat_map, obj_map, player, inventory):
        super().__init__(None, None, pos, mat_map, obj_map, inventory=inventory)
        self._player = player
        self.reset()

    def food_plan(self):
        print("--- Food plan ---")
        if self._inventory["steak"] > 0:
            self._sub_tasks = ["eat_steak"]
            print("Eat steak")
            return
        elif self._inventory["apple"] > 0:
            print("Eat apple")
            self._sub_tasks = ["eat_apple"]
            return
        elif self._inventory["beef"] > 0 and self._inventory["food"] < 3:
            print("Eat beef")
            self._sub_tasks = ["eat_beef"]
            return
        it = random.random()
        if it < 0.3:
            self._task_type, self._task = "get_task", "apple"
            print("Get apple")
        elif it < 0.6:
            self._task_type, self._task = "get_task", "beef"
            print("Get beef")
        else:
            self._task_type, self._task = "make_task", "steak"
            print("Get steak")
        self._sub_tasks = parse_task(self._task_type, self._task)
        self._sub_tasks.insert(0, "[food_plan")
        self._sub_tasks.append("eat_" + self._task)
        self._sub_tasks.append("]")
        return

    def drink_plan(self):
        print("--- Drink plan ---")
        if self._inventory["water_bucket"] > 0:
            self._sub_tasks = ["drink_" + "water_bucket"]
        if self._inventory["drink"] > 5:
            task_type, task = "get_task", "water"
            print("Get water with bucket")
            self._sub_tasks = parse_task(task_type, task)
        else:
            task_type, task = "get_task", "drink"
            self._sub_tasks = parse_task(task_type, task)
        self._sub_tasks.insert(0, "[drink_plan")
        self._sub_tasks.append("drink_" + task)
        self._sub_tasks.append("]")
        return

    def energy_plan(self):
        # total three policy:
        # 1. sleep
        # 2. make a bed, then sleep
        # 3. kill all zombies around, then sleep

        # if far from zombie and skeleton, sleep
        print("--- Energy plan ---")
        if random.random() < 0.5:
            # place a bed and sleep
            print("Place bed")
            self._task_type, self._task = "place_task", "bed"
            self._sub_tasks = parse_task(self._task_type, self._task)
            self._sub_tasks.append("sleep_bed")
        else:
            self._sub_tasks = ["sleep_none"]
        self._sub_tasks.insert(0, "[energy_plan")
        self._sub_tasks.append("]")
        return

    def weapon_plan(self):
        print("--- Weapon plan ---")
        task_type, task = "make_task", self._weapon
        self._sub_tasks = parse_task(task_type, task)
        return

    def sleep(self, where):
        # print(f"Sleep {where}")
        # print(self._player.inventory["energy"])
        if self._player.inventory["energy"] < 9:
            if where == "none":
                print("Sleep none")
                return "sleep", self._pos
            else:
                if self._mat_map[self._pos] == self._mat_ids["bed"]:
                    print("Sleep bed")
                    return "sleep", self._pos
                else:
                    pos = tuple(self._player.pos + self._player.facing)
                    self._path.insert(0, pos)
                    _, action, pos = self.move()
                    self._last_finished = False
                    return action, pos
        else:
            self._last_finished = True
            return "noop", self._pos
        return "sleep", self._pos

    def eat(self, food):
        self._last_finished = True
        return "eat_" + food, self._pos

    def drink(self, object = "none"):
        pos = tuple(self._player.pos + self._player.facing)
        self._last_finished = True
        if self._mat_map[pos] == self._mat_ids["water"]:
            print("Drink water directly")
            return "do", self._pos
        else:
            if self._inventory["water_bucket"] > 0:
                print("Drink water from bucket")
                return "drink_" + object, self._pos
        return "noop", self._pos

    def update_policy(self, state):
        # self.drink_plan()
        # return
        cache_list = np.array([state["food"], state["drink"], state["energy"]])
        intent = np.argmin(cache_list)
        if self.survival_flag == "no_food":
            cache_list = np.array([state["drink"], state["energy"]])
            intent = np.argmin(cache_list)
            if intent == 0:
                self.drink_plan()
            elif intent == 1:
                self.energy_plan()
        elif self.survival_flag == "no_drink":
            cache_list = np.array([state["food"], state["energy"]])
            intent = np.argmin(cache_list)
            if intent == 0:
                self.food_plan()
            elif intent == 1:
                self.energy_plan()
        elif self.survival_flag == "no_sleep":
            cache_list = np.array([state["food"], state["drink"]])
            intent = np.argmin(cache_list)
            if intent == 0:
                self.food_plan()
            elif intent == 1:
                self.drink_plan()
        elif self.survival_flag == "make_weapon":
            if self._weapon == None:
                self._weapon = random.choices(survival_constants.make_weapon)[0]
                # print(self._weapon)
            if self._player.inventory[self._weapon] > 0:
                self._last_finished = True
                nearest_zombie = self.find("zombie")
                nearest_skeleton = self.find("skeleton")
                pos = self._pos
                if isfar(pos, nearest_zombie) and isfar(pos, nearest_skeleton):
                    self._last_finished = True
                    self.energy_plan()
                else:
                    self._sub_tasks = parse_task("defeat_task", "zombie")
            else:
                self.weapon_plan()
        elif self.survival_flag == "all":
            cache_list = np.array([state["food"], state["drink"], state["energy"]])
            intent = np.argmin(cache_list)
            if intent == 0:
                self.food_plan()
            elif intent == 1:
                self.drink_plan()
            elif intent == 2:
                self.energy_plan()

    def action(self):
        if self._curr_task is None:
            self.update_policy(self._player.inventory)
        if len(self._sub_tasks) != 0:
            if self._last_finished:
                # check player state
                self._curr_task = self._sub_tasks.pop(0)
                while self._curr_task[0] in ["[", "]"]:
                    self.set_stack(self._curr_task)
                    if len(self._sub_tasks) == 0:
                        return "noop", self._pos
                    self._curr_task = self._sub_tasks.pop(0)
            action, target = self._curr_task.split('_', 1)
            func = getattr(self, action)
            return func(target)
            # print(f"--- Start from {self.start}, Destination {self.des}, now {self.pos} ---")
        else:
            self._curr_task = None
            return "noop", self._pos

    def update(self, mat_map, obj_map, agent_inventory):
        self._mat_map = mat_map
        self._obj_map = obj_map
        self._inventory = agent_inventory

    def reset(self):
        self.survival_flag = random.choices(survival_constants.survival_types)[0]
        print(f"Survival flag: {self.survival_flag}")
        self._curr_task = None
        self._sub_tasks = []
        self._path = []
        self._weapon = None
        # stack for subtasks info
        self._q_stack = []
        # list of dict {task, start, des} to save finished task information
        self._task_log = []