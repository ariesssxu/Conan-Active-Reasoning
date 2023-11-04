from tqdm import tqdm
from conan.gen.nav.agents import AstarAgent, NoisyAstarAgent
from conan.gen.nav.gen import task_init
from conan.playground import worldgen
from conan.playground import objects
from conan.playground import engine
from conan.playground import constants
import numpy as np
import collections
from conan.playground.env import Env
# from stable_baselines3.common.env_checker import check_env


class Nav_Env(Env):

    def __init__(self, area=(64, 64), view=(9, 11), size=(9, 9),
                 reward=True, length=10000, seed=None, boss=False, recover=False, footprints=False, view_type='symbolic'):
        super().__init__(area, view, size, reward, length, seed, boss, recover, footprints, view_type)

    def gen_path(self):
        # print(f"--- Gen nav ---")
        self.world._footprints = True
        mat_map, _ = self.get_detailed_view()
        start, des = task_init(self, mat_map)
        self._des = des
        agent = AstarAgent(start, des, mat_map)
        agent.get_path()

        running = True
        duration = 0

        while running:
            action, pos = agent.action()
            reward = 0
            done = False
            while tuple(self._player.pos) != pos:
                if self._world._obj_map[tuple(pos)] != 0:
                    self._world.remove(
                        self._world._objects[self._world._obj_map[tuple(pos)]])
                _, reward, done, _ = self.step(self.action_names.index(action))
            # env.player.pos = np.array(pos)
            if action == 'sleep':
                done = True
            duration += 1
            # Episode end.
            if done:
                # print('Gen path done, Duration:', duration)
                running = False
                self._world._footprints = False
                self._des = self._player.pos
                self._world.remove(self._player)
                self._world.add(objects.Player_bak(
                    self._world, self._des, alive=False))

    def reset(self):
        center = (self._world.area[0] // 2, self._world.area[1] // 2)
        self._episode += 1
        self._world.reset(
            seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
        self._update_time()
        self._player = objects.Player(self._world, center)
        self._last_health = self._player.health
        self._world.add(self._player)
        worldgen.generate_world(self._world, self._player)
        self.gen_path()
        self._player = objects.Player(self._world, center)
        if self._world._obj_map[center] != 0:
            self._world.remove(
                self._world._objects[self._world._obj_map[center]])
        self._world.add(self._player)
        self._step = 0
        self._unlocked = set()
        return self._obs()

    def step(self, action):
        self._step += 1
        self._update_time()
        self._player.action = constants.actions[action]
        for obj in self._world.objects:
            if self._player.distance(obj) < 2 * max(self._view):
                obj.update()
                obj.update_trace()
        if self._step % 10 == 0:
            if not self._recover:
                for chunk, objs in self._world.chunks.items():
                    # xmin, xmax, ymin, ymax = chunk
                    # center = (xmax - xmin) // 2, (ymax - ymin) // 2
                    # if self._player.distance(center) < 4 * max(self._view):
                    self._balance_chunk(chunk, objs)

        obs = self._obs()
        self._player.set_boss()

        dis = self._player.distance(self._des)
        # reward = -dis / 100
        reward = -0.01 if dis > 1 else 100

        finish = True if dis <= 1 else False
        over = self._length and self._step >= self._length
        done = finish or over
        info = {
            'inventory': self._player.inventory.copy(),
            'achievements': self._player.achievements.copy(),
            # 'discount': 1 - float(dead),
            'semantic': self._sem_view(),
            'player_pos': self._player.pos,
            'reward': reward,
        }
        if not self._reward:
            reward = 0.0
        return obs, reward, done, info