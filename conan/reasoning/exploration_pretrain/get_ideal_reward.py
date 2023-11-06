import argparse
import pygame
import conan.gen.nav.env as conan
from PIL import Image
import numpy as np
import gym
from conan.training.env import Nav_Env

reward_list = []

for i in range(10000):
    env = Nav_Env(size=(900, 1200), view=(9, 12), view_type='symbolic', length=1000)
    env.reset()
    reward_list.append(env._max_reward)

print(np.mean(reward_list))