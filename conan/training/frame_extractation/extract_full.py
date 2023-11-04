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


def read_ori_obs(file, frame_num):
    # obs_file = os.path.join(file, "obs.npy")
    npz_files = [x for x in os.listdir(file) if x.endswith('.npz')]
    npz_file = os.path.join(file, npz_files[0])
    npz = np.load(npz_file)
    frames = npz['mat_map']
    # sample frames
    frames_extracted = []
    for j in range(frame_num):
        frames_extracted.append(frames[(j * len(frames)) // frame_num])
    # idx = sorted(np.random.choice(frames.shape[0], frame_num, replace=False))
    # frames = frames[idx]
    return np.stack(frames_extracted)


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
    parser.add_argument('--save', default=os.path.join(pathlib.Path(__file__).parent, '../../dataset/dataset_full/goal/train'))
    args = parser.parse_args()

    dataset_dir = os.path.join(pathlib.Path(__file__).parent, '../../gen/save/task')

    config_file = os.path.join(args.save, 'config.json')

    configs = {}

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    task_dir = os.listdir(dataset_dir)
    random.shuffle(task_dir)
    print(len(task_dir))
    for eps in tqdm.tqdm(task_dir[:8000]):
        try:
            config = gen_config(os.path.join(dataset_dir, eps))
            if not config:
                continue
            images = read_ori_obs(os.path.join(dataset_dir, eps), 30)
            for idx, panel in enumerate(config):
                # images = np.array([])
                # save as npy
                np.save(os.path.join(args.save, f'{eps}_{idx}.npy'), images)
                configs[f"{eps}_{idx}"] = {"question": panel['question'], "choices": panel['choices'], "answer": panel['answer']}
        except:
            continue
    with open(config_file, 'w') as f:
        json.dump(configs, f)
