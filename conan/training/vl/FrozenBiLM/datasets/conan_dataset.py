import torch as th
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import pandas as pd
import pickle
import math
import json
import numpy as np
import os


class MC_Dataset(Dataset):
    def __init__(
        self,
        config_path,
        subtitles_path,
        feature_path,
        max_feats=30,
        features_dim=4096,
        tokenizer=None,
        use_context=True,
        type_map=None,
        prefix="",
        suffix="",
    ):
        f = open(config_path, "r")
        self.feature_path = feature_path
        self.data = json.load(f)
        if subtitles_path:
            self.subs = pickle.load(open(subtitles_path, "rb"))
        else:
            self.subs = None
        self.features = None
        self.max_feats = 30
        self.features_dim = features_dim
        self.mask = tokenizer.mask_token if tokenizer is not None else None
        self.use_context = use_context
        self.mc = 4
        self.type_map = type_map
        self.prefix = prefix
        self.suffix = suffix

    def __len__(self):
        return len(self.data)

    def _get_subtitles(self, video_id, start, end):
        # only consider subtitles that intersec with the timestamps of the video clip
        subs_list = [
            x["text"]
            for x in self.subs[video_id]
            if x["end"] >= start and x["start"] <= end
        ]
        return " ".join(subs_list).capitalize().strip()

    def _get_text(self, subtitles, answer, mask, question=None):
        text = (
            f"{self.prefix} Question: {question} Is it '{answer}'? {mask}{self.suffix}"
        )
        if self.use_context:
            text += f" Subtitles: {subtitles}"
        text = text.strip()
        return text

    def _get_video(self, idx):
        config = list(self.data.keys())[idx]
        if idx > len(self.data):
            video = th.zeros(1, self.features_dim)
        else:
            # load npy file
            video = np.load(os.path.join(
                self.feature_path, config + ".npy"))
            video = th.from_numpy(video)
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len,
                                 self.features_dim)], 0
            )
        else:
            video_len = self.max_feats
        # reshape to (max_feats, features_dim)
        video = video.view(self.max_feats, self.features_dim)
        return video, video_len

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
        # if "type" in self.data:
        #     type = self.data["type"].values[idx]

        # get subs
        # if self.subs:
        #     subs = self._get_subtitles(video_id, start, end)
        # else:
        #     subs = ""
        subs = ""
        # get features
        video, video_len = self._get_video(idx)

        # get answer id
        answer_id = config["answer"]

        text = []
        for i in range(self.mc):
            ai = config["choices"][i]
            text.append(self._get_text(subs, ai, self.mask, question))

        qid = idx
        if "qid" in self.data:
            qid = int(self.data["qid"].values[idx])

        return {
            "video": video,
            "video_len": video_len,
            "text": text,
            "qid": qid,
            "answer_id": answer_id,
            "type": type,
        }


def mc_collate_fn(batch):
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"]
                          for i in range(bs)], dtype=th.long)
    text = [
        [batch[i]["text"][j] for i in range(bs)] for j in range(len(batch[0]["text"]))
    ]
    qid = [batch[i]["qid"] for i in range(bs)]
    answer_id = default_collate([batch[i]["answer_id"] for i in range(bs)])
    type = [batch[i]["type"] for i in range(bs)]

    return {
        "video": video,
        "video_len": video_len,
        "text": text,
        "qid": qid,
        "answer_id": answer_id,
        "type": type,
    }


def build_mc_dataset(dataset_name, split, args, tokenizer):
    type_map = None
    if dataset_name == "conan":
        if split == "train":
            config_path = args.conan_train_config_path
        elif split == "val":
            config_path = args.conan_val_config_path
        elif split == "test":
            config_path = args.conan_val_config_path  # eval on val public
        else:
            raise NotImplementedError
        subtitles_path = args.conan_subtitles_path
    else:
        raise NotImplementedError
    feature_path = config_path.split("config")[0]
    return MC_Dataset(
        config_path=config_path,
        subtitles_path=subtitles_path,
        feature_path=feature_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        tokenizer=tokenizer,
        use_context=args.use_context,
        prefix=args.prefix,
        suffix=args.suffix,
        type_map=type_map,
    )
