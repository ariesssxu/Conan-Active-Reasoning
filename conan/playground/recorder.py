import datetime
import json
import pathlib

import imageio
import numpy as np
from PIL import Image


class Recorder:

    def __init__(
            self, env, record, directory, video_size=(900, 1100)):
        if record == "state":
            env = StatsRecorder(env, directory)
        elif record == "video":
            env = VideoRecorder(env, directory, video_size)
        elif record == "eps":
            env = EpisodeRecorder(env, directory)
        elif record == "all":
            env = AllRecorder(env, directory)
        self._env = env

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)


class StatsRecorder:

    def __init__(self, env, directory=None):
        self._env = env
        if directory:
            self._directory = pathlib.Path(directory).expanduser()
            self._directory.mkdir(exist_ok=True, parents=True)
            # self._file = (self._directory / 'stats.jsonl').open('a')
        self._length = None
        self._reward = None
        self._unlocked = None
        self._stats = None

    def set_directory(self, directory):
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        # self._file = (self._directory / 'stats.jsonl').open('a')

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)

    @property
    def env(self):
        return self._env

    def reset(self):
        obs = self._env.reset()
        self._length = 0
        self._reward = 0
        self._unlocked = None
        self._stats = None
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._length += 1
        self._reward += info['reward']
        if done:
            self._stats = {'length': self._length,
                           'reward': round(self._reward, 1)}
            for key, value in info['achievements'].items():
                self._stats[f'achievement_{key}'] = value
            self.save()
        return obs, reward, done, info

    def save(self, questions=None):
        pass
        # self._file.write(json.dumps(self._stats) + '\n')
        # self._file.flush()


class VideoRecorder(StatsRecorder):

    def __init__(self, env, directory, size=(900, 1100)):
        super().__init__(env, directory)
        if not hasattr(env, 'episode_name'):
            env = EpisodeName(env)
        self._env = env
        self._size = size
        self._frames = None

    def reset(self):
        obs = self._env.reset()
        self._frames = [self._env.render(self._size)]
        self._length = 0
        self._reward = 0
        self._unlocked = None
        self._stats = None
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._frames.append(self._env.render(self._size))
        self._length += 1
        self._reward += info['reward']
        if done:
            self._stats = {'length': self._length,
                           'reward': round(self._reward, 1)}
            for key, value in info['achievements'].items():
                self._stats[f'achievement_{key}'] = value
            self.save()
        return obs, reward, done, info

    def save(self, questions=None):
        self._file.write(json.dumps(self._stats) + '\n')
        self._file.flush()
        filename = str(self._directory / (self._env.episode_name + '.mp4'))
        imageio.mimsave(filename, self._frames)


class EpisodeRecorder:

    def __init__(self, env, directory):
        if not hasattr(env, 'episode_name'):
            env = EpisodeName(env)
        self._env = env
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        self._episode = None

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        self._episode = [{'image': obs}]
        return obs

    def step(self, action):
        # Transitions are defined from the environment perspective, meaning that a
        # transition contains the action and the resulting reward and next
        # observation produced by the environment in response to said action.
        obs, reward, done, info = self._env.step(action)
        transition = {
            'action': action, 'image': obs, 'reward': reward, 'done': done,
        }
        for key, value in info.items():
            if key in ('inventory', 'achievements'):
                continue
            transition[key] = value
        for key, value in info['achievements'].items():
            transition[f'achievement_{key}'] = value
        for key, value in info['inventory'].items():
            transition[f'ainventory_{key}'] = value
        self._episode.append(transition)
        if done:
            self.save()
        return obs, reward, done, info

    def save(self, questions=None):
        filename = str(self._directory / (self._env.episode_name + '.npz'))
        # Fill in zeros for keys missing at the first time step.
        for key, value in self._episode[1].items():
            if key not in self._episode[0]:
                self._episode[0][key] = np.zeros_like(value)
        episode = {
            k: np.array([step[k] for step in self._episode])
            for k in self._episode[0]}
        np.savez_compressed(filename, **episode)


class AllRecorder(VideoRecorder):

    def __init__(self, env, directory, save_video=False):
        super().__init__(env, directory)
        self._video = save_video

    def reset(self):
        self._env.reset()
        obs = self._env._obs_full()
        self._episode = [{'obs': obs}]
        self._frames = [self._env.render(self._size)]
        self._length = 0
        self._reward = 0
        self._unlocked = None
        self._stats = None
        return obs

    def step(self, action):
        # Transitions are defined from the environment perspective, meaning that a
        # transition contains the action and the resulting reward and next
        # observation produced by the environment in response to said action.
        _, reward, done, info = self._env.step(action)
        obs = self._env._obs_full()
        transition = {
            'action': action, 'obs': obs, 'reward': reward, 'done': done,
        }
        for key, value in info.items():
            if key in ('inventory', 'achievements'):
                continue
            transition[key] = value
        for key, value in info['achievements'].items():
            transition[f'achievement_{key}'] = value
        for key, value in info['inventory'].items():
            transition[f'ainventory_{key}'] = value
        transition['mat_map'], transition['obj_map'] = self._env.get_detailed_view()
        self._episode.append(transition)
        if self._video:
            self._frames.append(self._env.render(self._size))
            # img = self._env.render(self._size)
            # # save img as pdf
            # img = Image.fromarray(img)
            # img.save("test.pdf")
        self._length += 1
        self._reward += info['reward']
        if done:
            self._stats = {'length': self._length,
                           'reward': round(self._reward, 1)}
            for key, value in info['achievements'].items():
                self._stats[f'achievement_{key}'] = value
            self.save()
        return obs, reward, done, info

    def save(self, questions=None):
        filename = str(self._directory / (self._env.episode_name + '.npz'))
        # Fill in zeros for keys missing at the first time step.
        for key, value in self._episode[1].items():
            if key not in self._episode[0]:
                self._episode[0][key] = np.zeros_like(value)
        episode = {
            k: np.array([step[k] for step in self._episode])
            for k in self._episode[0]}
        np.savez_compressed(filename, **episode)
        # self._file.write(json.dumps(self._stats) + '\n')
        # self._file.flush()
        if self._video:
            filename = str(self._directory / (self._env.episode_name + '.mp4'))
            imageio.mimsave(filename, self._frames)
        if questions:
            filename = str(self._directory / (self._env.episode_name + '.json'))
            with open(filename, 'w') as f:
                json.dump(questions, f)


class EpisodeName:

    def __init__(self, env):
        self._env = env
        self._timestamp = None
        self._unlocked = None
        self._length = None

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        self._timestamp = None
        self._unlocked = None
        self._length = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._length += 1
        if done:
            self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            self._unlocked = sum(int(v >= 1)
                                 for v in info['achievements'].values())
        return obs, reward, done, info

    @property
    def episode_name(self):
        return f'{self._timestamp}-len{self._length}'
