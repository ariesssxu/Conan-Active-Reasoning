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

def read_ori_obs(file, frame_num):
    # obs_file = os.path.join(file, "obs.npy")
    npz_files = [x for x in os.listdir(file) if x.endswith('.npz')]
    npz_file = os.path.join(file, npz_files[0])
    npz = np.load(npz_file)
    frames = npz['obs']
    # sample frames
    frames_extracted = []

    for j in range(frame_num):
        frames_extracted.append(frames[(j * len(frames)) // frame_num])
    # idx = sorted(np.random.choice(frames.shape[0], frame_num, replace=False))
    # frames = frames[idx]

    return np.stack(frames_extracted)

def gen_key_frames(env, model, path, frame_num, step_num, render=False, mask=None):

    frames = []

    if render:
        pygame.init()
        screen = pygame.display.set_mode((900, 1200))
        size = [900, 1200]

    obs = env.reset(path, mask)

    for i in range(step_num):

        action, _states = model.predict(obs)

        obs, rewards, dones, info = env.step(action)

        obs_full = env._obs_full()
        if render:
            image = env.render(size)
            surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
        frames.append(obs_full)

    frames = np.array(frames)

    # sample frames
    frames_extracted = []
    for j in range(frame_num):
        frames_extracted.append(frames[(j * len(frames)) // frame_num])
    # idx = sorted(np.random.choice(frames.shape[0], frame_num, replace=False))
    # frames = frames[idx]
    return np.stack(frames_extracted)


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
    
    # question_file = os.path.join(file, "questions.json")
    question_file = os.path.join(file, "goal_questions.json")
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
    parser.add_argument('--explorer_save_path', default=os.path.join(pathlib.Path(__file__).parent, '../../dataset_trpo_2/goal/val'))
    parser.add_argument('--obs_save_path', default=os.path.join(pathlib.Path(__file__).parent, '../../dataset_obs_new_2/goal/val'))    
    parser.add_argument('--obs', type=bool, default=True) 
    args = parser.parse_args()

    ckpt_path = f'{args.outdir}/conan_5e7'
    env = Nav_Env(size=(900, 1200), view=(9, 12), view_type='symbolic', length=1000)
    dataset_dir = os.path.join(pathlib.Path(__file__).parent, '../../gen/save/task')
    env.save_path = dataset_dir
    model = load_model(env, ckpt_path)

    
    # with open(config_file, 'r') as f:
    #     configs = json.load(f)
    # print(len(configs))

    explorer_config_file = os.path.join(args.explorer_save_path, 'config.json')
    configs = {}
    if not os.path.exists(args.explorer_save_path):
        os.makedirs(args.explorer_save_path)
    
    if args.obs:
        obs_config_file = os.path.join(args.obs_save_path, 'config.json')
        if not os.path.exists(args.obs_save_path):
            os.makedirs(args.obs_save_path)

    task_dir = os.listdir(dataset_dir)
    
    # your own seed here
    random.Random(4).shuffle(task_dir)
    print(len(task_dir))
    
    for eps in tqdm.tqdm(task_dir[9000:]):
        try:
            config = gen_config(os.path.join(dataset_dir, eps))
            if not config:
                continue
            if args.obs:
                obs_images = read_ori_obs(os.path.join(dataset_dir, eps), 30)
            for idx, panel in enumerate(config):
                explorer_images = gen_key_frames(env, model, eps, 30, 150, mask=panel['mask'])
                
                # images = np.array([])
                # save as npy
                np.save(os.path.join(args.explorer_save_path, f'{eps}_{idx}.npy'), explorer_images)
                if args.obs:
                    np.save(os.path.join(args.obs_save_path, f'{eps}_{idx}.npy'), obs_images)
                configs[f"{eps}_{idx}"] = {"question": panel['question'], "choices": panel['choices'], "answer": panel['answer']}
        except Exception as e:
            print(e)
            continue
        
    with open(explorer_config_file, 'w') as f:
        json.dump(configs, f)
        
    if args.obs:
        with open(obs_config_file, 'w') as f:
            json.dump(configs, f)
