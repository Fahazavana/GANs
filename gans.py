import platform
import torchvision
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


class Discriminator(nn.Module):
    """
    Discriminator model
    """

    def __init__(self, img_dim=748):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
                nn.Linear(img_dim, 128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    """
    Generator model
    """

    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
                nn.Linear(z_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, img_dim),
                nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


def get_device():
    if platform.platform().lower().startswith("mac"):  # macOS
        return "mps" if torch.backends.mps.is_available() else "cpu"
    else:  # Linux, Windows
        return "cuda" if torch.cuda.is_available() else "cpu"


def train(gen, disc, latent_dim, train_loader, optimizer, criterion, epochs, device='cpu'):
    gen.train(True)
    disc.train(True)
    N = len(train_loader)
    M = len(train_loader.dataset)
    FIXED_NOISE = torch.randn((4, latent_dim)).to(device)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    with torch.no_grad():
        generator.eval()
        fake_imgs = generator(FIXED_NOISE)
        fake_imgs = fake_imgs.view(-1, 1, 28, 28).cpu()
        print(fake_imgs.shape)
        img_grid_fake = torchvision.utils.make_grid(fake_imgs, nrow=2,normalize=True)
        print(img_grid_fake.shape)
        # plot = ax.imshow(fake_imgs[0].permute(1, 2, 0), cmap='gray')
        plot = ax.imshow(img_grid_fake.permute(1,2,0), cmap='gray')
        plt.axis('off')
        plt.show(block=False)
        # exit()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(enumerate(train_loader), total=N, desc=f"Epoch {epoch}/{epochs}")
        run_dloss, run_gloss = 0, 0
        for k, (real_imgs, _) in pbar:
            real_imgs = real_imgs.view(-1, 28 * 28 * 1).to(device)
            # noise
            z = torch.randn(real_imgs.shape[0], latent_dim).type_as(real_imgs)
            ########################################################################
            # generator  training max log (D(G(z)))
            fake_imgs = generator(z)
            y_hat = discriminator(fake_imgs)

            y_fake = torch.ones(real_imgs.shape[0], 1)
            y_fake = y_fake.type_as(real_imgs)
            g_loss = criterion(y_hat, y_fake)

            optimizer[0].zero_grad()
            g_loss.backward()
            optimizer[1].step()
            run_gloss += g_loss.item()*real_imgs.shape[0]

            ########################################################################
            # discriminator training max log D(x) + log ( 1- D(G(z)))

            y_hat_real = discriminator(real_imgs)
            y_real = torch.ones(real_imgs.shape[0], 1).type_as(real_imgs)
            real_loss = criterion(y_hat_real, y_real)

            z = generator(z).detach()
            y_hat_fake = discriminator(z)
            y_fake = torch.zeros(real_imgs.shape[0], 1).type_as(real_imgs)
            fake_loss = criterion(y_hat_fake, y_fake)
            d_loss = (real_loss + fake_loss) / 2

            optimizer[1].zero_grad()
            d_loss.backward()
            optimizer[0].step()
            run_dloss += d_loss.item()*real_imgs.shape[0]
            # Update the progress bar
            pbar.set_postfix(g_loss=f"{run_gloss / M:.3f}", d_loss=f"{run_dloss / M:.3f}")

        with torch.no_grad():
            generator.eval()
            fake_imgs = generator(FIXED_NOISE)
            fake_imgs = fake_imgs.view(-1, 1, 28, 28).cpu()
            img_grid_fake = torchvision.utils.make_grid(fake_imgs, nrow=2, normalize=True)
            # plot = ax.imshow(fake_imgs[0].permute(1, 2, 0), cmap='gray')
            plot = ax.imshow(img_grid_fake.permute(1,2,0), cmap='gray')
            plot.figure.canvas.draw()
            plt.pause(0.001)


if __name__ == "__main__":
    device = get_device()
    ROOT = "../DATASET/MNIST/train/"
    LR = 2e-4
    BATCH_SIZE = 256*2
    Z_DIM = 28*28
    IMAGE_SIZE = 28 * 28 * 1
    EPOCHS = 50

    discriminator = Discriminator(IMAGE_SIZE).to(device)
    generator = Generator(Z_DIM, IMAGE_SIZE).to(device)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    train_dataset = MNIST(ROOT, train=True, transform=transform, download=False)

    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = [optim.AdamW(discriminator.parameters(), lr=LR),
                 optim.AdamW(generator.parameters(), lr=LR)]

    train(generator, discriminator, Z_DIM, dataloader, optimizer, nn.BCELoss(), EPOCHS, get_device())