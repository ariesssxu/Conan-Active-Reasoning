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

def gen_key_frames(env, path, frame_num, step_num, render=False, mask=None):
    return np.zeros((frame_num, 64, 64))


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
    parser.add_argument('--nenvs', type=int, default=1)
    parser.add_argument('--explorer_save_path', default=os.path.join(pathlib.Path(__file__).parent, '../../dataset/dataset_empty/goal/test'))
    args = parser.parse_args()

    env = Nav_Env(size=(900, 1200), view=(9, 12), view_type='symbolic', length=1000)
    # dataset_dir = os.path.join(pathlib.Path(__file__).parent, '../../gen/save/task')
    dataset_dir = os.path.join(pathlib.Path(__file__).parent, '../../gen/save/task')
    env.save_path = dataset_dir

    explorer_config_file = os.path.join(args.explorer_save_path, 'config.json')
    configs = {}
    if not os.path.exists(args.explorer_save_path):
        os.makedirs(args.explorer_save_path)

    task_dir = os.listdir(dataset_dir)
    
    # your own seed here
    random.Random(32).shuffle(task_dir)
    print(len(task_dir))
    
    for eps in tqdm.tqdm(task_dir[7000:]):
        try:
            config = gen_config(os.path.join(dataset_dir, eps))
            if not config:
                continue
            for idx, panel in enumerate(config):
                explorer_images = gen_key_frames(env, eps, 30, 1, mask=panel['mask'])
                
                # images = np.array([])
                # save as npy
                np.save(os.path.join(args.explorer_save_path, f'{eps}_{idx}.npy'), explorer_images)
                configs[f"{eps}_{idx}"] = {"question": panel['question'], "choices": panel['choices'], "answer": panel['answer']}
        except Exception as e:
            print(e)
            continue
        
    with open(explorer_config_file, 'w') as f:
        json.dump(configs, f)
        
