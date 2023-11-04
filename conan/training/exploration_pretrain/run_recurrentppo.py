import argparse
import pygame
import conan.gen.nav.env as conan
import stable_baselines3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from PIL import Image
import numpy as np
import gym
import time
from sb3_contrib import RecurrentPPO
from conan.training.env import Nav_Env
import os
import pathlib


def vis_test(path):
    pygame.init()
    screen = pygame.display.set_mode((900, 1200))

    env = Nav_Env(size=(900, 1200), view=(9, 12), view_type='symbolic', length=1000)

    model = stable_baselines3.DQN(
        'MlpPolicy',
        env,
        verbose=1,
        batch_size=int(args.bs),
        buffer_size=int(args.buffer_size)
    )

    model.load(path)
    obs = env.reset()
    size = [900, 1200]
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        image = env.render(size)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        # time.sleep(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=os.path.join(pathlib.Path(__file__).parent, '../ckpt/rl/conan_nav_recurrentppo/'))
    parser.add_argument('--steps', type=float, default=1e8)
    parser.add_argument('--nenvs', type=int, default=4)
    parser.add_argument('--buffer_size', type=int, default=1e6)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--tensorboard-log', type=str, default=os.path.join(pathlib.Path(__file__).parent, '../tensorboard_log/conan_nav_recurrentppo/'))
    args = parser.parse_args()

    # vis_test(path=f'{args.outdir}/conan_remote')
    # return

    env = make_vec_env('conan_nav-v0', n_envs=args.nenvs, vec_env_cls=SubprocVecEnv)

    # model = stable_baselines3.PPO('CnnPolicy', env, verbose=1, batch_size=args.bs)
    # use Mlp for symbolic obs and CNN for visual input.
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        tensorboard_log=args.tensorboard_log
    )

    model.learn(
        total_timesteps=args.steps,
        tb_log_name="recurrentppo-5000"
    )
    model.save(f'{args.outdir}/conan_tmp')
    # tb_log_name="diamond_limited_eps_500_bz_128_new_obs"
    # model.save(f'{args.outdir}/conan_nav_dqn_500')

    # del model  # remove to demonstrate saving and loading
