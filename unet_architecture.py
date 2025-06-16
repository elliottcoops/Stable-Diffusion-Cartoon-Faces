import torch
import torch.nn as nn
import math

# Sinusoidal time embedding for diffusion timesteps
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        """
        timesteps: (B,) long tensor of time step indices
        returns: (B, embedding_dim) float tensor of time embeddings
        """
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]  # (B, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# Small UNet model with time embeddings, adapted for latent space input
class UNetWithTimeEmbedding(nn.Module):
    def __init__(self,latent_channels,time_emb_dim=128,text_emb_dim=512):
        super().__init__()

        self.config = type('', (), {})()  # empty dummy config object
        self.config._diffusers_version = "0.9.0"
    
        self.text_proj = nn.Sequential(
            nn.Linear(text_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)

        # Map time embeddings to a vector for conditional modulation
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Encoder conv blocks
        self.encoder1 = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )

        # Decoder conv blocks
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        # Final conv layer to predict noise
        self.final_conv = nn.Conv2d(64, latent_channels, 1)

        # To condition conv layers on time embeddings (simple way: add after each block)
        self.time_emb_proj1 = nn.Linear(time_emb_dim, 64)
        self.time_emb_proj2 = nn.Linear(time_emb_dim, 128)
        self.time_emb_proj_bottleneck = nn.Linear(time_emb_dim, 256)

    def forward(self, x, t, text=None):
        # t emb
        t_emb = self.time_embedding(t)  # (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb)     # (B, time_emb_dim)
    
        if text is not None:
            text = self.text_proj(text)  # (B, time_emb_dim)
            t_emb = t_emb + text

        # Encoder 1
        x1 = self.encoder1(x)  # (B,64,H,W)
        # Add time embedding as bias broadcasted to channels & spatial dims
        x1 = x1 + self.time_emb_proj1(t_emb)[:, :, None, None]
        p1 = self.pool1(x1)

        # Encoder 2
        x2 = self.encoder2(p1)  # (B,128,H/2,W/2)
        x2 = x2 + self.time_emb_proj2(t_emb)[:, :, None, None]
        p2 = self.pool2(x2)

        # Bottleneck
        b = self.bottleneck(p2)  # (B,256,H/4,W/4)
        b = b + self.time_emb_proj_bottleneck(t_emb)[:, :, None, None]

        # Decoder 2
        up2 = self.up2(b)
        concat2 = torch.cat([up2, x2], dim=1)
        d2 = self.decoder2(concat2)

        # Decoder 1
        up1 = self.up1(d2)
        concat1 = torch.cat([up1, x1], dim=1)
        d1 = self.decoder1(concat1)

        return self.final_conv(d1)