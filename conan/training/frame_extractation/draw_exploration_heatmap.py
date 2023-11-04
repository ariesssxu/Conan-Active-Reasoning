# load
import pygame
import conan.gen.nav.env as conan
import stable_baselines3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from PIL import Image
import numpy as np
import os
import gym
import argparse
import time
from conan.training.env import Nav_Env
import pathlib
import tqdm
import json
import random
from sb3_contrib import TRPO
from sb3_contrib import RecurrentPPO
import seaborn as sns
import matplotlib.pyplot as plt


def gen_key_frames(env, model, path, frame_num, step_num, render=False, mask=None):

    frames = []

    if render:
        pygame.init()
        screen = pygame.display.set_mode((900, 1200))
        size = [900, 1200]

    obs = env.reset("get_diamond_8", mask=np.zeros((64, 64)))

    traces = np.ones((64, 64))

    for i in range(step_num):

        action, _states = model.predict(obs)

        obs, rewards, dones, info = env.step(action)

        # obs_full = (env._obs_full() > 0).astype(np.int)
        # traces += obs_full
        pos = env.player.pos
        print(pos)
        traces[tuple(pos)] += 3
        if render:
            image = env.render(size)
            surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

    return np.log(traces)


def load_model(env, ckpt_path):
    # model = stable_baselines3.DQN(
    #     'MlpPolicy',
    #     env,
    #     verbose=1,
    #     device='cuda',
    # )

    model = TRPO(
        'MlpPolicy',
        env,
        verbose=1,
    )

    model.load(ckpt_path)
    return model


def gen_config(file):

    question_file = os.path.join(file, "questions.json")
    # question_file = os.path.join(file, "goal_questions.json")
    if not os.path.exists(question_file):
        return None
    f = open(question_file, 'r')
    questions = json.load(f)
    return questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=os.path.join(pathlib.Path(__file__).parent, '../ckpt/rl/conan_nav_trpo'))
    parser.add_argument('--steps', type=float, default=2e8)
    parser.add_argument('--nenvs', type=int, default=1)
    parser.add_argument('--buffer_size', type=int, default=1e7)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--obs', type=bool, default=True)
    args = parser.parse_args()

    ckpt_path = f'{args.outdir}/conan_5e7'
    env = Nav_Env(size=(6400, 6500), view=(9, 12), view_type='symbolic', length=1000)
    dataset_dir = os.path.join(pathlib.Path(__file__).parent, '../../gen/save/diamond_tmp')
    env.save_path = dataset_dir
    model = load_model(env, ckpt_path)

    # with open(config_file, 'r') as f:
    #     configs = json.load(f)
    # print(len(configs))

    task_dir = os.listdir(dataset_dir)

    # your own seed here
    random.Random(4).shuffle(task_dir)
    print(len(task_dir))

    path = ""
    explorer_images = gen_key_frames(env, model, path, 30, 150, mask=np.ones([64, 64]))
    sns.heatmap(explorer_images, square=True, cbar=False)
    plt.xlim(0, explorer_images.shape[0])
    plt.ylim(0, explorer_images.shape[1])
    plt.axis('off')
    plt.savefig("heatmap.png", bbox_inches='tight', pad_inches=0)
