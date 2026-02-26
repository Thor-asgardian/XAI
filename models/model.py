import torch.nn as nn
from models.encoders import ModalityEncoder
from models.fusion import AttentionFusion

class MXMDF(nn.Module):
    def __init__(self, input_dims, latent_dim):
        super().__init__()

        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(dim, latent_dim)
            for name, dim in input_dims.items()
        })

        self.fusion = AttentionFusion(latent_dim, len(input_dims))

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        embeddings = []
        for name, encoder in self.encoders.items():
            embeddings.append(encoder(inputs[name]))

            fused, attn = self.fusion(embeddings)
            output = self.classifier(fused)
            return output, attn