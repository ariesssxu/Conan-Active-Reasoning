import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import einops
from typing import Optional, Tuple, Union
from conan.training.vl.head import Head, UNet


class FBiLM_HEAD(nn.Module):
    def __init__(self, lm_model, model_name, head_path=None):

        super().__init__()

        self.lm_model = lm_model
        self.device = self.lm_model.device
        self.model_name = model_name

        # # [batch_size, 30, 64, 64] to [batch_size, 30, 64, 64]
        # self.head = nn.Sequential(
        #     nn.Flatten(),               # flatten each image in the batch
        #     nn.Linear(30 * 64 * 64, 1024),  # fully connected layer with 1024 output units
        #     nn.ReLU(),                  # apply ReLU activation
        #     nn.Linear(1024, 30 * 64 * 64),  # fully connected layer with 30*64*64 output units
        #     nn.Sigmoid(),               # apply sigmoid activation
        #     nn.Unflatten(1, (30, 64, 64))  # reshape the output back to [batch_size, 30, 64, 64]
        # )
        # self.head = UNet(n_channels=30, n_classes=48)
        self.head = Head()
        if head_path:
            self.head.load_state_dict(torch.load(head_path))
            print(f"Load head from {head_path}")
        
    
    def freeze_lm(self):
        for param in self.lm_model.parameters():
            param.requires_grad = False
        self.lm_model.eval()
        print("freeze lm model")
        
    def freeze_head(self):
        for param in self.head.parameters():
            param.requires_grad = False
        self.head.eval()

    def forward(self,
            video=None,
            video_mask=None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            mlm=False
        ):
        video = einops.rearrange(video, 'b c (d e) -> b c d e', d=64, e=64)
        video = self.head(video)
        video = einops.rearrange(video, 'b c d e -> b c (d e)')
        if "bert" in self.model_name and "deberta" not in self.model_name:
            return self.lm_model(video,
                video_mask,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                encoder_hidden_states,
                encoder_attention_mask,
                labels,
                output_attentions,
                output_hidden_states,
                return_dict,
                mlm)
        elif "deberta" in self.model_name:
            return self.lm_model(input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                inputs_embeds,
                labels,
                output_attentions,
                return_dict,
                video,
                video_mask,
                mlm,
            )
            
        else:
            raise NotImplementedError
        return None

    def set_answer_embeddings(self, a2tok, freeze_last=True):
        return self.lm_model.set_answer_embeddings(a2tok, freeze_last=True)
