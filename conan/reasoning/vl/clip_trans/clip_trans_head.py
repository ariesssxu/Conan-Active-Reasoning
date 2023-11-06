import pytorch_lightning as pl
from clip_trans_module import CLIPTransWOTextEncoder
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torchmetrics
from conan.training.vl.head import Head, UNet

task_names = ["None"]


class CLIPTransHEAD(pl.LightningModule):
    def __init__(self, clip_path=None, head_path=None):

        super().__init__()

        # self.clip_model = CLIPTransWOTextEncoder.load_from_checkpoint(clip_path)
        if not clip_path:
            self.clip_model = CLIPTransWOTextEncoder(clip_model='ViT-B-16.pt', d_model=512,
                                    num_layers=6, nhead=8, dim_feedforward=2048, n_classes=5, dropout=0.1, context_length=64) 
        else:
            CLIPTransWOTextEncoder.load_from_checkpoint(clip_path)

        self.head = Head()
        if head_path:
            self.head.load_state_dict(torch.load(head_path))
            print(f"Load head from {head_path}")
            
        # [batch_size, 30, 64, 64] to [batch_size, 30, 64, 64]
        # self.head = nn.Sequential(
        #     nn.Flatten(),               # flatten each image in the batch
        #     nn.Linear(30 * 64 * 64, 1024),  # fully connected layer with 1024 output units
        #     nn.ReLU(),                  # apply ReLU activation
        #     nn.Linear(1024, 30 * 64 * 64),  # fully connected layer with 30*64*64 output units
        #     nn.Sigmoid(),               # apply sigmoid activation
        #     nn.Unflatten(1, (30, 64, 64))  # reshape the output back to [batch_size, 30, 64, 64]
        # )

        self.train_accs = nn.ModuleList([torchmetrics.Accuracy(
            task='multiclass', num_classes=5) for _ in range(len(task_names))])
        self.val_accs = nn.ModuleList([torchmetrics.Accuracy(
            task='multiclass', num_classes=5) for _ in range(len(task_names))])
        self.test_accs = nn.ModuleList([torchmetrics.Accuracy(
            task='multiclass', num_classes=5) for _ in range(len(task_names))])

    
    def freeze_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()
        print("freeze clip")
        
    def freeze_head(self):
        for param in self.head.parameters():
            param.requires_grad = False
        self.head.eval()
        
    def forward(self, images, texts):
        images = self.head(images)
        return self.clip_model(images, texts)

    def training_step(self, batch, batch_idx):
        images, texts, labels, task_idx = batch
        logits = self(images, texts)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss, sync_dist=True)

        for i in range(len(task_names)):
            task_mask = (task_idx == i)
            if task_mask.sum() > 0:
                task_logits = logits[task_mask]
                task_labels = labels[task_mask]
                acc = self.train_accs[i](task_logits, task_labels)

        return loss

    def validation_step(self, batch, batch_idx):
        images, texts, labels, task_idx = batch
        logits = self(images, texts)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss, sync_dist=True)

        for i in range(len(task_names)):
            task_mask = (task_idx == i)
            if task_mask.sum() > 0:
                task_logits = logits[task_mask]
                task_labels = labels[task_mask]
                acc = self.val_accs[i](task_logits, task_labels)

        return loss

    def test_step(self, batch, batch_idx):
        images, texts, labels, task_idx = batch
        logits = self(images, texts)
        loss = F.cross_entropy(logits, labels)
        self.log('test_loss', loss, sync_dist=True)

        for i in range(len(task_names)):
            task_mask = (task_idx == i)
            if task_mask.sum() > 0:
                task_logits = logits[task_mask]
                task_labels = labels[task_mask]
                acc = self.test_accs[i](task_logits, task_labels)

        return loss

    def on_training_epoch_end(self, *arg, **kwargs):
        for i in range(len(task_names)):
            self.log(
                f'train_acc_{task_names[i]}', self.train_accs[i].compute(), sync_dist=True)

    def on_validation_epoch_end(self, *arg, **kwargs):
        for i in range(len(task_names)):
            self.log(f'val_acc_{task_names[i]}',
                     self.val_accs[i].compute(), sync_dist=True)

    def on_test_epoch_end(self, *arg, **kwargs):
        for i in range(len(task_names)):
            self.log(
                f'test_acc_{task_names[i]}', self.test_accs[i].compute(), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=0.01)
        return optimizer
