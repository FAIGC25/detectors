import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        x: [B, SeqLen, D] (Output from VideoMAE)
        Returns:
            weighted_features: [B, D] (Для классификации)
            attn_weights: [B, SeqLen] (Для ГРАФИКОВ в ДЕМО!)
        """
        scores = self.attn(x)

        attn_weights = F.softmax(scores, dim=1)

        weighted_features = torch.sum(x * attn_weights, dim=1) # [B, D]

        return weighted_features, attn_weights.squeeze(-1)