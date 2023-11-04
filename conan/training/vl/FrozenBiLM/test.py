import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import math
import sys
from typing import Iterable
import argparse
import time
import datetime
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from collections import namedtuple
from functools import reduce

from datasets import build_mc_dataset, mc_collate_fn
from model import build_model, get_tokenizer
from main import get_args_parser
from util.misc import get_mask, adjust_learning_rate, mask_tokens
from util.metrics import MetricLogger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Frozen training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    # args.model_name = os.path.join(
    #     os.environ["TRANSFORMERS_CACHE"], args.model_name)

    mc_dataset = build_mc_dataset("conan", "val", args, tokenizer=None)
    print(mc_dataset[0])
    
