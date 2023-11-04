import collections

import numpy as np

from . import constants
from . import engine
from . import objects
from . import worldgen


# Gym is an optional dependency.
try:
    import gym
    DiscreteSpace = gym.spaces.Discrete
    BoxSpace = gym.spaces.Box
    DictSpace = gym.spaces.Dict
    BaseClass = gym.Env
except ImportError:
    DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
    BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
    DictSpace = collections.namedtuple('DictSpace', 'spaces')
    BaseClass = object


class Env(BaseClass):

    def __init__(
            self, area=(64, 64), view=(9, 12), size=(64, 64),
            reward=True, length=100, seed=None, boss=False, recover=False, footprints=True, other_agents=True, view_type='symbolic'):
        view = np.array(view if hasattr(view, '__len__') else (view, view))
        size = np.array(size if hasattr(size, '__len__') else (size, size))
        seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
        self._area = area
        self._view = view
        self._size = size
        self._reward = reward
        self._length = length
        self._seed = seed
        self._boss = boss
        self._footprints = footprints
        self._recover = recover
        self._episode = 0
        self._view_type = view_type
        self._other_agents = other_agents
        self._step = 0
        self._player = None
        self._last_health = None
        self._unlocked = None
        # Some libraries expect these attributes to be set.
        self.reward_range = None
        self.metadata = None
        self._world = engine.World(
            area, constants.materials, (16, 16), self._footprints, self._other_agents, seed=self._seed)
        self._textures = engine.Textures(constants.root / 'assets')
        item_rows = int(np.ceil(len(constants.items) / view[0]))
        if view[0] != view[1] - item_rows:
            print(f"WARNING: Length {view[0]}!= Height {view[1] - item_rows}!")
        self._obs_size = tuple([view[0], view[1] - item_rows])
        self._local_view = engine.LocalView(
            self._world, self._textures, [view[0], view[1] - item_rows])
        self._item_view = engine.ItemView(
            self._textures, [view[0], item_rows])
        self._sem_view = engine.SemanticView(self._world, [
            getattr(objects, x.capitalize()) for x in constants.objects])
        self._sym_view = engine.SymbolicView(
            tuple(self._obs_size))
        self._detailed_view = engine.DetailedView(self._world, [
            getattr(objects, x.capitalize()) for x in constants.objects])
        self._sym_full_view = engine.SymbolicFullView(
            tuple(self._obs_size))

    @ property
    def world(self):
        return self._world

    @ property
    def player(self):
        return self._player

    def _obs_full(self):
        return self._sym_full_view(self._player.pos, self._sem_view())

    @ property
    def observation_space(self):
        if self._view_type == 'visual':
            return BoxSpace(0, 255, tuple(self._size) + (3,), np.uint8)
        else:
            return BoxSpace(0, 255, self._obs_size, np.uint8)

    @ property
    def action_space(self):
        return DiscreteSpace(len(constants.actions))

    @ property
    def player(self):
        return self._player

    @ property
    def action_names(self):
        return constants.actions

    def get_detailed_view(self):
        return self._detailed_view()

    def reset(self):
        center = (self._world.area[0] // 2, self._world.area[1] // 2)
        self._episode += 1
        self._step = 0
        self._world.reset(
            seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
        self._update_time()
        self._player = objects.Player(self._world, center)
        self._last_health = self._player.health
        self._world.add(self._player)
        self._unlocked = set()
        worldgen.generate_world(self._world, self._player)
        return self._obs()

    def step(self, action):
        self._step += 1
        self._update_time()
        self._player.action = constants.actions[action]
        for obj in self._world.objects:
            if self._player.distance(obj) < 2 * max(self._view):
                obj.update()
                obj.update_trace()
        # if self._step % 10 == 0:
        #     if not self._recover:
        #         for chunk, objs in self._world.chunks.items():
        #             # xmin, xmax, ymin, ymax = chunk
        #             # center = (xmax - xmin) // 2, (ymax - ymin) // 2
        #             # if self._player.distance(center) < 4 * max(self._view):
        #             self._balance_chunk(chunk, objs)
        obs = self._obs()

        if self._boss:
            self._player.set_boss()

        # original reward
        reward = (self._player.health - self._last_health) / 10
        self._last_health = self._player.health
        unlocked = {
            name for name, count in self._player.achievements.items()
            if count > 0 and name not in self._unlocked}
        if unlocked:
            self._unlocked |= unlocked
            reward += 1.0

        dead = self._player.health <= 0
        over = self._length and self._step >= self._length
        done = dead or over
        info = {
            'inventory': self._player.inventory.copy(),
            'achievements': self._player.achievements.copy(),
            'discount': 1 - float(dead),
            'semantic': self._sem_view(),
            'player_pos': self._player.pos,
            'reward': reward,
        }
        if not self._reward:
            reward = 0.0
        return obs, reward, done, info

    def render(self, size=None):
        size = size or self._size
        unit = size // self._view
        canvas = np.zeros(tuple(size) + (3,), np.uint8)
        local_view = self._local_view(self._player, unit)
        item_view = self._item_view(self._player.inventory, unit)
        view = np.concatenate([local_view, item_view], 1)
        border = (size - (size // self._view) * self._view) // 2
        (x, y), (w, h) = border, view.shape[:2]
        canvas[x: x + w, y: y + h] = view
        return canvas.transpose((1, 0, 2))

    def _obs(self):
        if self._view_type == "visual":
            return self.render()
        else:
            return self._sym_view(self._player.pos, self._sem_view())

    def _update_time(self):
        # https://www.desmos.com/calculator/grfbc6rs3h
        progress = (self._step / 300) % 1 + 0.3
        daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3
        self._world.daylight = daylight

    def _balance_chunk(self, chunk, objs):
        light = self._world.daylight
        self._balance_object(
            chunk, objs, objects.Zombie, 'grass', 6, 0, 0.01, 0.01,
            lambda pos: objects.Zombie(self._world, pos, self._player),
            lambda num, space: (
                0 if space < 50 else 3.5 - 3 * light, 3.5 - 3 * light))
        self._balance_object(
            chunk, objs, objects.Skeleton, 'path', 7, 7, 0.01, 0.01,
            lambda pos: objects.Skeleton(self._world, pos, self._player),
            lambda num, space: (0 if space < 6 else 1, 2))
        self._balance_object(
            chunk, objs, objects.Cow, 'grass', 5, 5, 0.01, 0.01,
            lambda pos: objects.Cow(self._world, pos),
            lambda num, space: (0 if space < 30 else 1, 1.5 + light))

    def _balance_object(
            self, chunk, objs, cls, material, span_dist, despan_dist,
            spawn_prob, despawn_prob, ctor, target_fn):
        xmin, xmax, ymin, ymax = chunk
        random = self._world.random
        creatures = [obj for obj in objs if isinstance(obj, cls)]
        mask = self._world.mask(*chunk, material)
        target_min, target_max = target_fn(len(creatures), mask.sum())
        if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
            xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
            ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
            xs, ys = xs[mask], ys[mask]
            i = random.randint(0, len(xs))
            pos = np.array((xs[i], ys[i]))
            empty = self._world[pos][1] is None
            away = self._player.distance(pos) >= span_dist
            if empty and away:
                self._world.add(ctor(pos))
        elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
            obj = creatures[random.randint(0, len(creatures))]
            away = self._player.distance(obj.pos) >= despan_dist
            if away:
                self._world.remove(obj)
