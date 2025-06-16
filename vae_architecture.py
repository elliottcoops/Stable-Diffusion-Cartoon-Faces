import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, latent_channels=4):
        super().__init__()
        # convolutional layers similar to before but designed to reduce spatial dims
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 4, 2, 1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, 4, 2, 1)  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, 2, 1)  # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(hidden_channels * 4, hidden_channels * 8, 4, 2, 1)  # 8x8 -> 4x4
        
        # Instead of flattening, output conv layers for mean and logvar
        self.conv_mean = nn.Conv2d(hidden_channels * 8, latent_channels, 3, 1, 1)
        self.conv_logvar = nn.Conv2d(hidden_channels * 8, latent_channels, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  
        
        mean = self.conv_mean(x)    
        log_var = self.conv_logvar(x)  

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_channels=4, hidden_channels=64, out_channels=3):
        super().__init__()
        # Starting from spatial latent tensor
        self.deconv1 = nn.ConvTranspose2d(latent_channels, hidden_channels * 8, 4, 2, 1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, 4, 2, 1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, 2, 1)  # 16x16 -> 32x32
        self.deconv4 = nn.ConvTranspose2d(hidden_channels * 2, out_channels, 4, 2, 1)  # 32x32 -> 64x64

    def forward(self, z):
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = torch.sigmoid(self.deconv4(z))
        
        return z


class SpatialVAE(nn.Module):
    def __init__(self, latent_channels=4, hidden_channels=64, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, latent_channels)
        self.decoder = Decoder(latent_channels, hidden_channels, out_channels)

    def forward(self, x):
        mean, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std  # reparameterization on spatial latents
        reconstructed = self.decoder(z)
        return reconstructed, mean, log_var
