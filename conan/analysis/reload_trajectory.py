# load saved trajectory from all recorder npz files

import argparse
import os

import numpy as np
try:
    import pygame
except ImportError:
    print('Please install the pygame package to use the GUI.')
    raise
from PIL import Image

import conan.playground as playground

if __name__ == '__main__':
    def boolean(x): return bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
    # arg2 = arg1 + 1/2
    parser.add_argument('--view', type=int, nargs=2, default=(9, 11))
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--health', type=int, default=9)
    parser.add_argument('--window', type=int, nargs=2, default=(900, 1100))
    parser.add_argument('--size', type=int, nargs=2, default=(0, 0))
    parser.add_argument('--save_path', type=str, default="save")
    parser.add_argument('--record', type=str, default='state', choices=[
        'state', 'video', 'eps', 'all'])
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--wait', type=boolean, default=False)
    parser.add_argument('--boss', type=boolean, default=False)
    parser.add_argument('--death', type=str, default='quit', choices=[
        'continue', 'reset', 'quit'])
    args = parser.parse_args()

    playground.constants.items['health']['max'] = args.health
    playground.constants.items['health']['initial'] = args.health

    size = list(args.size)
    size[0] = size[0] or args.window[0]
    size[1] = size[1] or args.window[1]

    env = playground.Env(
        area=args.area, view=args.view, length=args.length, seed=args.seed, boss=args.boss)
    env.reset()
    # achievements = set()
    # duration = 0
    # return_ = 0
    # was_done = False
    # print('Diamonds exist:', env._world.count('diamond'))

    size = list(args.size)
    size[0] = size[0] or args.window[0]
    size[1] = size[1] or args.window[1]

    pygame.init()
    screen = pygame.display.set_mode(args.window)
    clock = pygame.time.Clock()

    # load npz file
    log_path = 'save/'
    npz_files = [x for x in os.listdir(log_path) if x.endswith('.npz')]
    npz_file = os.path.join(log_path, npz_files[0])
    npz = np.load(npz_file)
    for i in range(len(npz["mat_map"])):
        playground.worldgen.load_world(env._world, env._player, npz["mat_map"][i], npz["obj_map"][i])
        image = env.render(size)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
    