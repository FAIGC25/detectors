import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCEConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, nce_logits, target_weights):
        """
        nce_logits: [B, SeqLen] (Cosine Similarity / Temperature)
        target_weights: [B, SeqLen] (Attention Weights from Pooler)
        """
        log_probs = F.log_softmax(nce_logits, dim=1)
        loss = -torch.sum(target_weights.detach() * log_probs, dim=1).mean()
        return loss