# generate path with astar

import argparse
import numpy as np
import pygame
from PIL import Image
import conan.playground as playground
import os
import numpy as np
from conan.gen.nav.astar_solver import AstarSolver
from tqdm import trange
from random import randint
import time
from conan.gen.nav.agents import AstarAgent, NoisyAstarAgent


def task_init(env, mat_map, start=None):
    # start
    des = [randint(0, 63), randint(0, 63)]
    if not start:
        start = [randint(0, 63), randint(0, 63)]
    else:
        start = list(start)
    if astar_nav(mat_map, start, des) != None:
        if env.world._obj_map[tuple(start)] == 0:
            # due to chunk, cannot directly set player.pos
            env.world.move(env.player, np.array(start))
            return tuple(start), tuple(des)
        elif env.world._obj_map[tuple(start)] == 1:
            return tuple(start), tuple(des)
    return task_init(env, mat_map, start)
    # return des


def astar_nav(mat_map, start, des):
    Solver = AstarSolver(mat_map)
    way = []
    possible = Solver.astar(tuple(start), tuple(des))
    if possible:
        way = list(possible)
    else:
        way = None
    return way


def gen_astar_single(env, args, size, id):

    env = playground.Recorder(env, args.record, args.save_path, args.render)
    env.reset()
    achievements = set()
    duration = 0
    return_ = 0
    was_done = False

    print(f"--- Gen nav naive {id} ---")

    mat_map, _ = env.env.get_detailed_view()
    start, des = task_init(env, mat_map)
    # print(start, des)
    # agent = AstarAgent(start, des, mat_map)
    agent = NoisyAstarAgent(start, des, mat_map)
    agent.get_path()
    pygame.init()

    if args.render:
        screen = pygame.display.set_mode(args.window)

    clock = pygame.time.Clock()
    running = True

    while running:

        action, pos = agent.action()
        # Environment step.
        reward = 0
        done = False
        while tuple(env.player.pos) != pos:
            _, reward, done, _ = env.step(env.action_names.index(action))
        # env.player.pos = np.array(pos)
        if action == 'sleep':
            done = True
        duration += 1

        # Rendering.
        if args.render:
            image = env.render(size)
            if size != args.window:
                image = Image.fromarray(image)
                image = image.resize(args.window, resample=Image.NEAREST)
                image = np.array(image)
            surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            # time.sleep(0.1)

        # Episode end.
        if done and not was_done:
            was_done = True
            print('Episode done!')
            print('Duration:', duration)
            if args.death == 'quit':
                running = False


def gen_astar(args):

    playground.constants.items['health']['max'] = args.health
    playground.constants.items['health']['initial'] = args.health

    size = list(args.size)
    size[0] = size[0] or args.window[0]
    size[1] = size[1] or args.window[1]

    env = playground.Env(
        area=args.area,
        view=args.view,
        length=args.length,
        seed=args.seed,
        boss=args.boss,
    )

    for i in trange(args.num):
        gen_astar_single(env, args, size, i)


def main():
    def boolean(x): return bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
    # arg2 = arg1 + 1/2
    parser.add_argument('--view', type=int, nargs=2, default=(9, 12))
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--health', type=int, default=9)
    parser.add_argument('--window', type=int, nargs=2, default=(900, 1100))
    parser.add_argument('--size', type=int, nargs=2, default=(0, 0))
    parser.add_argument('--save_path', type=str, default="save/nav/1")
    parser.add_argument('--record', type=str, default='state', choices=[
        'state', 'video', 'eps', 'all'])
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--boss', type=boolean, default=True)
    parser.add_argument('--death', type=str, default='quit', choices=[
        'continue', 'reset', 'quit'])
    parser.add_argument('--footprints', type=boolean, default=True)
    parser.add_argument('--render', type=boolean, default=True)
    args = parser.parse_args()

    gen_astar(args)


if __name__ == '__main__':
    main()
