# ====================================================================================================================
# Using Audio Model. Dont use
# ====================================================================================================================


from base import BaseAVDFModule
import torch
import torch.nn as nn
from src.pooler.base_pooler import NormalizeLearningPooler
from src.backbones.av_former import AVFormerPretrained

class AVDFRecPretrained(BaseAVDFModule):
    def __init__(self, pretrained_model_name: str = "bert-base-uncased", lora_cfg=None,
                 hidden_size: int = 512,
                 num_classes: int = 2,
                 **kwargs):
        super().__init__(**kwargs)


        self.av_transformer = AVFormerPretrained(
            model_name=pretrained_model_name,
            lora_cfg=lora_cfg,
            embed_dim=self.d_model
        )
        self.pooler = NormalizeLearningPooler(num_latents=self.num_latents)
        pooler_out_dim = self.num_latents * self.num_latents

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooler_out_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, video_inputs, audio_inputs, fau_inputs):
        q, z_a, z_v = self.extract_features(video_inputs, audio_inputs, fau_inputs)

        z_features = torch.cat([z_a, z_v], dim=1)

        outputs = self.av_transformer(q, z_features)
        full_sequence = outputs.last_hidden_state
        out_av = full_sequence[:, :self.num_latents, :]

        pool_av = self.pooler(out_av, out_av)
        logits = self.classifier(pool_av)
        return logits