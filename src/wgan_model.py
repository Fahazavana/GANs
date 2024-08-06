import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=5, bias=False), # 16x16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # 8x8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # 4x4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.75),
            nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=0, bias=False), # 1,1
            nn.Sigmoid()
            )


    def forward(self, x):
        x = self.disc(x)
        return x.view(-1,1)



class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.noise2fc = nn.Linear(in_features=latent_dim, out_features=7 * 7 * 64)
        self.cnvt = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.noise2fc(x)
        x = x.view(-1, 64, 7, 7)
        x = self.cnvt(x)
        return x
        