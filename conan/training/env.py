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
import os
import random
import gym
import json
# from stable_baselines3.common.env_checker import check_env

DiscreteSpace = gym.spaces.Discrete
BoxSpace = gym.spaces.Box


class Nav_Env(Env):

    def __init__(self, area=(64, 64), view=(9, 12), size=(9, 9),
                 reward=True, length=300, seed=None, boss=False, recover=False, footprints=False, view_type='symbolic', pre_load=False):
        super().__init__(area, view, size, reward, length, seed, boss, recover, footprints, view_type)
        self._pre_load = pre_load
        self.save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../gen/save/diamond_tmp")
        if self._pre_load:
            self.pre_load_paths()

    # for explorator
    @ property
    def action_space(self):
        return DiscreteSpace(len(constants.actions[:7]))

    @ property
    def observation_space(self):
        return BoxSpace(0, 255, (2, 64, 64), np.uint8)

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

    def load_path(self, path=None):
        # print(path)
        if not self._pre_load:
            path_dir = random.choice(os.listdir(self.save_path)) if not path else path
            tmp_dir = os.path.join(self.save_path, path_dir)
            if not os.path.isdir(tmp_dir) and not path:
                self.load_path(path)
                return
            elif not os.path.isdir(tmp_dir) and path:
                print("Wrong path!", tmp_dir)
                return
            npz_files = [x for x in os.listdir(tmp_dir) if x.endswith('.npz')]
            # if len(npz_files) == 0 or not npz_files[0].startswith("None"):
            if len(npz_files) == 0:
                # print(f"--- No npz files in {tmp_dir} ---")
                # self.load_path(path)
                print("Wrong path!", tmp_dir)
                return
            npz_file = os.path.join(tmp_dir, npz_files[0])
            json_file = npz_file[:-4] + ".json"
            f = open(json_file, 'r')
            self._config = json.load(f)
            npz = np.load(npz_file)
            # for i in range(len(npz["mat_map"])):
            worldgen.load_world(self._world, self._player, npz["mat_map"][-1], npz["obj_map"][-1])
        else:
            i = random.randint(0, len(self._paths) - 1)
            mat_map, obj_map = self._paths[i]
            worldgen.load_world(self._world, self._player, mat_map, obj_map)

    def pre_load_paths(self):
        print("Start preload")
        self._paths = []
        for path_dir in os.listdir(self.save_path):
            tmp_dir = os.path.join(self.save_path, path_dir)
            if not os.path.isdir(tmp_dir):
                continue
            npz_files = [x for x in os.listdir(tmp_dir) if x.endswith('.npz')]
            if len(npz_files) == 0 or not npz_files[0].startswith("None"):
                continue
            npz_file = os.path.join(tmp_dir, npz_files[0])
            npz = np.load(npz_file)
            self._paths.append((npz["mat_map"][-1], npz["obj_map"][-1]))

    def cal_reward(self, obs):
        traces_set_1 = self._traces_set_1 + [self._obj_ids["player_bak"]]
        traces_set_2 = self._traces_set_2 + [self._obj_ids["player_bak"]]
        self._rewarded_array_1 = np.logical_or(self._rewarded_array_1, np.isin(obs, traces_set_1)).astype(float)
        self._rewarded_array_2 = np.logical_or(self._rewarded_array_2, np.isin(obs, traces_set_2)).astype(float)
        reward = np.sum(self._rewarded_array_1) + 2 * np.sum(self._rewarded_array_2)
        return reward

    def prepare_reward(self):
        mat_map, _ = self.get_detailed_view()
        rewarded_array_1 = np.isin(mat_map, self._traces_set_1).astype(float)
        rewarded_array_2 = np.isin(mat_map, self._traces_set_2).astype(float)
        reward = np.sum(rewarded_array_1) + 2 * np.sum(rewarded_array_2)
        self._max_reward = reward
        self._rewarded_array_1 = np.zeros_like(mat_map)
        self._rewarded_array_2 = np.zeros_like(mat_map)
        # print(f"--- Max reward: {self._max_reward} ---")

    def gen_mask(self):
        start = self._config[0]["start"][1:-1].split(", ")
        des = self._config[-1]["des"][1:-1].split(", ")
        mask = [[min(int(start[0]), int(des[0])), min(int(start[1]), int(des[1]))], [max(int(start[0]), int(des[0])), max(int(start[1]), int(des[1]))]]
        self.set_mask(mask)

    def set_mask(self, mask):
        mask_64 = np.zeros((64, 64))
        mask_64[int(mask[0][0]):int(mask[1][0]) + 1, int(mask[0][1]):int(mask[1][1]) + 1] = 1
        self._mask = mask_64

    def reward(self, obs):
        reward = self.cal_reward(obs)
        return False, reward

    def reset(self, path=None, mask=None):
        center = (self._world.area[0] // 2, self._world.area[1] // 2)
        self._episode += 1
        self._world.reset(
            seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
        self._update_time()

        # gen_path: randomly generate path, load_path: load existing traces
        # self.gen_path()
        self._obj_ids = {
            c: len(self.world._mat_ids) + i
            for i, c in enumerate(constants.objects)}
        self._traces_set_1 = [self.world._mat_ids[x] for x in constants.traces_1]
        self._traces_set_2 = [self.world._mat_ids[x] for x in constants.traces_2]
        self._rewarded = set()
        self._total_reward = 0
        self._player = objects.Player(self._world, center)

        self.load_path(path)
        self.prepare_reward()

        if mask is not None:
            self.set_mask(mask)
        else:
            self.gen_mask()

        if self._world._obj_map[center] != 0:
            self._world.remove(
                self._world._objects[self._world._obj_map[center]])
        self._world.add(self._player)
        self._step = 0
        self._unlocked = set()

        obs_full = self._obs_full()
        obs = np.concatenate([[obs_full], [self._mask]], axis=0)
        return obs

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

        obs_full = self._obs_full()
        obs = np.concatenate([[obs_full], [self._mask]], axis=0)
        self._player.set_boss()

        # dis = self._player.distance(self._des)
        # reward = -0.01 if dis > 1 else 100
        # finish = True if dis <= 1 else False

        # here is the total reward
        finish, reward = self.reward(obs_full)
        current_reward = reward - self._total_reward
        self._total_reward = reward

        # step penalty
        current_reward -= 0.01
        if action == 5:
            current_reward -= 1

        over = self._length and self._step >= self._length
        if self._total_reward >= self._max_reward:
            over = True
            current_reward += 100
        done = finish or over
        info = {
            'inventory': self._player.inventory.copy(),
            'achievements': self._player.achievements.copy(),
            # 'discount': 1 - float(dead),
            'semantic': self._sem_view(),
            'player_pos': self._player.pos,
            'reward': current_reward,
        }
        if not self._reward:
            reward = 0.0
        return obs, current_reward, done, info


if __name__ == '__main__':
    env = Nav_Env()
    obs = env.reset()
    print(obs.shape)
    # check_env(env)
