import torch
import pathlib
import torchvision
import numpy as np
import clip
import os
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import json
import pytorch_lightning as pl

import datasets


class MC_Dataset(Dataset):
    def __init__(self, dataset_path, context_length=77):
        f = open(os.path.join(dataset_path, "config.json"), "r")
        self.dataset_path = dataset_path
        self.context_length = context_length
        self.data = json.load(f)
        self.mc = 4
        # self.prefix = prefix
        # self.suffix = suffix
        self.prefix = '<image>'
        self.eoc = '<EOC>'
        self.max_feats = 30

    def __len__(self):
        return len(self.data)

    def _clip_tokenize(self, text):
        return clip.tokenize(text, context_length=self.context_length)

    def _get_text(self, subtitles, answer, question=None):
        text = (
            f"{self.prefix} Question: {question} Answer: {answer}"
        )
        if self.use_context:
            text += f" Subtitles: {subtitles}"
        text = text.strip()
        return text

    def _get_video(self, idx):
        config = list(self.data.keys())[idx]
        if idx > len(self.data):
            video = torch.zeros(1, self.features_dim)
        else:
            # load npy file
            video = np.load(os.path.join(
                self.dataset_path, config + ".npy"))
            video = torch.from_numpy(video)
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat(
                [video, torch.zeros(self.max_feats - video_len,
                                    self.features_dim)], 0
            )
        else:
            video_len = self.max_feats
        # reshape to (max_feats, features_dim)
        # video = video.view(self.max_feats, self.features_dim)
        return video.float(), video_len

    def __getitem__(self, idx):

        config = self.data[list(self.data.keys())[idx]]

        # get start, end
        # start = self.data["start"].values[idx]
        # end = self.data["end"].values[idx]

        # get question
        question = config["question"]
        if question[-1] != "?":
            question = str(question) + "?"
        type = 0
        # get features
        video, video_len = self._get_video(idx)

        # get answer id
        answer_id = config["answer"]

        text = f"{question}"
        for i in range(self.mc):
            ai = config["choices"][i]
            text += f"{ai} "

        text = self._clip_tokenize(text)

        return video, text, answer_id, 0


class ConanCLIPData(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=32, num_workers=4, context_length=77, **kwargs):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = MC_Dataset(os.path.join(
            self.dataset_path, "train"), context_length=context_length)
        self.val_dataset = MC_Dataset(
            os.path.join(self.dataset_path, "val"), context_length=context_length)
        self.test_dataset = MC_Dataset(
            os.path.join(self.dataset_path, "test"), context_length=context_length)

    def train_dataloader(self):
        # distributed
        if self.trainer.num_devices > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset)
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=train_sampler)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        # distributed
        if self.trainer.num_devices > 1:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset)
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=val_sampler)
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # distributed
        if self.trainer.num_devices > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset)
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=test_sampler)
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    data = MC_Dataset(
        'dataset/survival/val', context_length=64)
    loader = DataLoader(data, batch_size=16, num_workers=4, shuffle=True)

    # decode and print
    from clip.simple_tokenizer import SimpleTokenizer as st
    tokenizer = st()
    for batch in loader:
        for i in range(16):
            print(tokenizer.decode(batch[1][i][0].tolist()))
        pass
