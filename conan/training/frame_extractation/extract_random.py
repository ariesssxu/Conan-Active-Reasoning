# load
import pygame
import conan.gen.nav.env as conan
from PIL import Image
import numpy as np
import os
import argparse
import time
from conan.training.env import Nav_Env
import pathlib
import tqdm
import json
import random

def gen_key_frames(env, path, frame_num, step_num, render=False, mask=None):
    
    frames = []
    
    if render:
        pygame.init()
        screen = pygame.display.set_mode((900, 1200))
        size = [900, 1200]

    obs = env.reset(path, mask)

    for i in range(step_num):

        action = random.choice([0, 1, 2, 3, 4])

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
    parser.add_argument('--nenvs', type=int, default=1)
    parser.add_argument('--explorer_save_path', default=os.path.join(pathlib.Path(__file__).parent, '../../dataset/dataset_random_exploration/intent/val'))
    args = parser.parse_args()

    env = Nav_Env(size=(900, 1200), view=(9, 12), view_type='symbolic', length=1000)
    dataset_dir = os.path.join(pathlib.Path(__file__).parent, '../../gen/save/task')
    env.save_path = dataset_dir

    explorer_config_file = os.path.join(args.explorer_save_path, 'config.json')
    configs = {}
    if not os.path.exists(args.explorer_save_path):
        os.makedirs(args.explorer_save_path)

    task_dir = os.listdir(dataset_dir)
    
    # your own seed here
    random.Random(4).shuffle(task_dir)
    print(len(task_dir))
    
    for eps in tqdm.tqdm(task_dir[9000:]):
        try:
            config = gen_config(os.path.join(dataset_dir, eps))
            if not config:
                continue
            for idx, panel in enumerate(config):
                explorer_images = gen_key_frames(env, eps, 30, 150, mask=panel['mask'])                
                np.save(os.path.join(args.explorer_save_path, f'{eps}_{idx}.npy'), explorer_images)
                configs[f"{eps}_{idx}"] = {"question": panel['question'], "choices": panel['choices'], "answer": panel['answer']}
        except Exception as e:
            print(e)
            continue
        
    with open(explorer_config_file, 'w') as f:
        json.dump(configs, f)
        
