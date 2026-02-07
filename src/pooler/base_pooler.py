import torch.nn as nn
import torch


class NormalizeLearningPooler(nn.Module):
    def __init__(self, num_latents: int):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape=num_latents)
        self.par = nn.Parameter(data=torch.ones(1))

    def forward(self, z_aq, z_vq):

        mp = z_aq @ z_vq.transpose(1,2)

        return self.norm(self.par * mp)