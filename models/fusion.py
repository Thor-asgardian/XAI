import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, latent_dim, num_modalities):
        super().__init__()
        self.attn = nn.Linear(latent_dim, 1)

    def forward(self, embeddings):
        stack = torch.stack(embeddings, dim=1)
        scores = torch.softmax(self.attn(stack), dim=1)
        fused = torch.sum(stack * scores, dim=1)
        return fused, scores.squeeze(-1)