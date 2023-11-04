from conan_dataset_clip import ConanCLIPData
from clip_trans_module import CLIPTrans, CLIPTransWOTextEncoder
from clip_trans_head import CLIPTransHEAD
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import argparse
import os

TASK = "intent"


def get_args_parser():
    parser = argparse.ArgumentParser("Set FrozenBiLM", add_help=False)

    # Task
    parser.add_argument(
        "--task",
        default="intent",
        choices=["intent", "goal", "survival"],
    )

    # Model ckpt
    parser.add_argument(
        "--ckpt",
        default=None,
    )

    # head
    parser.add_argument(
        "--head",
        action='store_true',
        # default=False,
    )

    # dataset
    parser.add_argument(
        "--obs_dataset",
        action='store_true',
    )

    # save dir
    parser.add_argument(
        "--save_dir",
        default="",
    )

    return parser


if __name__ == "__main__":
    # module = CLIPTrans(clip_model='ViT-B-16.pt')

    args = get_args_parser().parse_args()
    print(args.head, args.obs_dataset)
    module = CLIPTransWOTextEncoder(clip_model='ViT-B-16.pt', d_model=512,
                                    num_layers=6, nhead=8, dim_feedforward=2048, n_classes=5, dropout=0.1, context_length=64) \
        if not args.head else CLIPTransHEAD(args.ckpt)
    # module = CLIPTransHEAD()
    default_root_dir = f"{args.task}" + args.save_dir

    dataset_path = f'dataset_trpo/{args.task}/' if not args.obs_dataset \
        else f'dataset_obs/{args.task}/'

    trainer = Trainer(accelerator="gpu", devices=1, num_nodes=1, max_epochs=100,
                      gradient_clip_val=1.0, check_val_every_n_epoch=5, log_every_n_steps=10,
                      default_root_dir=default_root_dir)

    trainer.fit(module, datamodule=ConanCLIPData(
        dataset_path=dataset_path, batch_size=128, context_length=64))

    # test
    trainer.test(module, datamodule=ConanCLIPData(
        dataset_path=dataset_path, batch_size=64, context_length=64))
