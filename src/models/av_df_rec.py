# ====================================================================================================================
# Using Audio Model. Dont use
# ====================================================================================================================

from base import BaseAVDFModule
import torch
import torch.nn as nn
from src.backbones.av_former import AVFormer, QSharedFormer
from src.pooler.base_pooler import NormalizeLearningPooler

class AVDFRec(BaseAVDFModule):
    def __init__(self, transformer_type: str = "AVFormer", hidden_size: int = 512,
                 num_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        if transformer_type == "AVFormer":
            self.av_transformer = AVFormer(d_model_out=self.d_model)
        elif transformer_type == "QSharedFormer":
            self.av_transformer = QSharedFormer(d_model=self.d_model)
        else:
            raise ValueError("Unknown dual-stream transformer type")

        self.pooler = NormalizeLearningPooler(num_latents=self.num_latents)
        pooler_out_dim = 3 * (self.num_latents * self.num_latents)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooler_out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3*num_classes)
        )

    def forward(self, video_inputs, audio_inputs, fau_inputs):
        q, z_a, z_v = self.extract_features(video_inputs, audio_inputs, fau_inputs)

        out_a, out_v = self.av_transformer(q, z_a, z_v)

        pool_aa = self.pooler(out_a, out_a)
        pool_vv = self.pooler(out_v, out_v)
        pool_va = self.pooler(out_v, out_a)

        combined = torch.cat([
            pool_aa.flatten(1),
            pool_vv.flatten(1),
            pool_va.flatten(1)
        ], dim=1)


        logits = self.classifier(combined)

        aa_logits, vv_logits, va_logits = torch.split(logits, logits.shape[1] // 3, dim=1)
        return aa_logits, vv_logits, va_logits