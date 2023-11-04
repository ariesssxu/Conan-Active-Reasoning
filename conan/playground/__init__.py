from .env import Env
from .recorder import Recorder

try:
  import gym
  gym.register(
      id='playgroundReward-v1',
      entry_point='playground:Env',
      max_episode_steps=10000,
      kwargs={'reward': True})
  gym.register(
      id='playgroundNoReward-v1',
      entry_point='playground:Env',
      max_episode_steps=10000,
      kwargs={'reward': False})
except ImportError:
  pass
