from .modeling_flamingo import FlamingoModelForMultipleChoice
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torchmetrics
from conan.training.vl.head import Head, UNet


class FlamingoHEAD(nn.Module):
    def __init__(self, config, ckpt_path=None, head_path=None):

        super().__init__()

        if not ckpt_path:
            self.flamingo_model = FlamingoModelForMultipleChoice(config)   
        else:
            self.flamingo_model = FlamingoModelForMultipleChoice.from_pretrained(ckpt_path)
        
        self.head = Head()
        # # [batch_size, 30, 64, 64] to [batch_size, 30, 64, 64]
        # self.head = nn.Sequential(
        #     nn.Flatten(),               # flatten each image in the batch
        #     nn.Linear(30 * 64 * 64, 1024),  # fully connected layer with 1024 output units
        #     nn.ReLU(),                  # apply ReLU activation
        #     nn.Linear(1024, 30 * 64 * 64),  # fully connected layer with 30*64*64 output units
        #     nn.Sigmoid(),               # apply sigmoid activation
        #     nn.Unflatten(1, (30, 64, 64))  # reshape the output back to [batch_size, 30, 64, 64]
        # )
        
        # self.head = Head.load_state_dict(torch.load(head_path))
        # print(f"Load head from {head_path}")

    def freeze_flamingo(self):
        for param in self.flamingo_model.parameters():
            param.requires_grad = False
        self.flamingo_model.eval()
        print("freeze flamingo")
        
    def freeze_head(self):
        for param in self.head.parameters():
            param.requires_grad = False
        self.head.eval()
        
    def forward(
        self,
        input_ids= None,
        attention_mask = None,
        media_locations = None,
        pixel_values = None, # N T c h w
        visual_features = None,
        head_mask = None,
        inputs_embeds = None,
        use_cache = False,
        past_key_values = None,
        return_dict = True,
        labels = None,
        loss_reduction = 'mean',
        **kwargs
    ):
        pixel_values = self.head(pixel_values)
        # print(pixel_values.shape)
        return self.flamingo_model(
            input_ids,
            attention_mask,
            media_locations,
            pixel_values, # N T c h w
            visual_features,
            head_mask,
            inputs_embeds,
            use_cache,
            past_key_values,
            return_dict,
            labels,
            loss_reduction,
            **kwargs
        )
