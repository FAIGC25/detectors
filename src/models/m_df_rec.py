# ====================================================================================================================
# Using Audio Model. Dont use
# ====================================================================================================================


import torch.nn as nn
from base import BaseAVDFModule
from src.backbones.av_former import MixedFormer, QSharedMixedFormer
from src.pooler.base_pooler import NormalizeLearningPooler

class AVDFRecMixed(BaseAVDFModule):
    def __init__(self, transformer_type: str = "MixedFormer", hidden_size: int = 512,
                 num_classes: int = 2, **kwargs):
        super().__init__(**kwargs)

        if transformer_type == "MixedFormer":
            self.av_transformer = MixedFormer(d_model=self.d_model)
        elif transformer_type == "QSharedMixedFormer":
            self.av_transformer = QSharedMixedFormer(d_model=self.d_model)
        else:
            raise ValueError("Unknown mixed transformer type")

        pooler_out_dim = 1 * (self.num_latents * self.num_latents)
        self.pooler = NormalizeLearningPooler(num_latents=self.num_latents)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooler_out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, video_inputs, audio_inputs, fau_inputs):
        q, z_a, z_v = self.extract_features(video_inputs, audio_inputs, fau_inputs)
        out_av = self.av_transformer(q, z_a, z_v)
        pool_av = self.pooler(out_av, out_av)

        logits = self.classifier(pool_av)
        return logits