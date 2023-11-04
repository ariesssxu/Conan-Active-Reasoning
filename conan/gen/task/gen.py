# generate random tasks

import argparse
import numpy as np
import pygame
from PIL import Image
from conan import playground
import random
import os
import numpy as np
from tqdm import trange
import conan.gen.task.task_constants as task_constants
from conan.gen.task.agents import TaskAgent
from conan.gen.task.check import task_check, cando_check


def task_init(env, mat_map, task):
    if task:
        task_type = task.split("_")[0] + "_task"
        task = "_".join(task.split("_")[1:])
        return task_type, task
    else:
        task_type = random.choice(
            ["get_task", "defeat_task", "make_task", "place_task"])
        return task_type, random.choice(getattr(task_constants, task_type))


def gen_task_single(env, args, size, id, task_type, task):

    env.reset()
    duration = 0

    print(f"----- Gen task trace {id} -----")

    mat_map, obj_map = env.env.get_detailed_view()
    # task_type, task = "get_task", "diamond"
    task_name = task_type.split("_")[0] + "_" + task
    save_path = os.path.join(args.save_path, f"{task_name}_{id}")
    env.set_directory(save_path)
    print(f"----- Task: {task_type} {task} -----")

    # return

    # agent = AstarAgent(start, des, mat_map)
    agent = TaskAgent(task_type, task, env.player.pos, mat_map, obj_map)
    agent.parse_task()
    # print(agent._sub_tasks)

    pygame.init()
    if args.render:
        screen = pygame.display.set_mode(args.window)
    clock = pygame.time.Clock()
    running = True
    done = False
    mat_map, obj_map = env.env.get_detailed_view()

    while running:

        action, pos = agent.action()
        if action == "reset":
            return False

        # at least one step
        _, reward, done, _ = env.step(env.action_names.index(action))
        mat_map, obj_map = env.env.get_detailed_view()
        index = 0
        while tuple(env.player.pos) != pos:
            _, reward, done, _ = env.step(env.action_names.index(action))
            index += 1
            if index > 20:
                return False
            if cando_check(mat_map, obj_map, pos):
                action_new = "do"
                _, reward, done, _ = env.step(
                    env.action_names.index(action_new))
            mat_map, obj_map = env.env.get_detailed_view()
        # env.player.pos = np.array(pos)
        # if action == 'sleep':
        #     done = True
        agent.update(mat_map, obj_map, env.player.inventory)
        finished = task_check(task_type, task, env.env)
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
        if done or finished:
            if finished:
                if len(agent._q_stack) > 0:
                    agent.set_stack("]")
                env.save(questions=agent._task_log)
            print('Episode done!')
            print('Duration:', duration)
            if args.death == 'quit':
                running = False
            return finished


def gen_tasks(args):

    playground.constants.items['health']['max'] = args.health
    playground.constants.items['health']['initial'] = args.health

    size = list(args.size)
    size[0] = size[0] or args.window[0]
    size[1] = size[1] or args.window[1]

    for finished in trange(0, args.num):
        env = playground.Env(
            area=args.area,
            view=args.view,
            length=args.length,
            seed=args.seed,
            boss=args.boss,
            other_agents=args.other_agents,
        )

        env = playground.Recorder(
            env, args.record, args.save_path, args.save_video)
        done = False
        mat_map, obj_map = env.env.get_detailed_view()
        task_type, task = task_init(env, mat_map, args.task)

        # try:
        iter = 0
        while not done and iter < 20:
            env.reset()
            mat_map, obj_map = env.env.get_detailed_view()
            done = gen_task_single(env, args, size, finished, task_type, task)
            iter += 1
            if iter == 20:
                print(f"Iter 20 still not done, task {task_type} {task}")
        # except Exception as e:
        #     print(e)
        #     print(f"Task {task_type} {task} failed")
        #     continue

        finished += 1
        done = False


def main():

    def boolean(x): return bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
    # arg2 = arg1 + 1/2
    parser.add_argument('--view', type=int, nargs=2, default=(9, 12))
    parser.add_argument('--length', type=int, default=300)
    parser.add_argument('--health', type=int, default=9)
    parser.add_argument('--window', type=int, nargs=2, default=(900, 1200))
    parser.add_argument('--size', type=int, nargs=2, default=(0, 0))
    parser.add_argument('--save_path', type=str, default="save/task/")
    parser.add_argument('--save_video', type=boolean, default=True)
    parser.add_argument('--record', type=str, default='all', choices=[
        'state', 'video', 'eps', 'all'])
    parser.add_argument('--fps', type=int, default=50)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--boss', type=boolean, default=True)
    parser.add_argument('--other_agents', type=boolean, default=True)
    parser.add_argument('--death', type=str, default='quit', choices=[
        'continue', 'reset', 'quit'])
    parser.add_argument('--footprints', type=boolean, default=True)
    parser.add_argument('--render', type=boolean, default=True)
    parser.add_argument('--task', default=None)
    args = parser.parse_args()

    gen_tasks(args)


if __name__ == '__main__':
    main()
