import torch.nn as nn

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.model(x)