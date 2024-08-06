import platform
import torchvision
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from IPython.display import clear_output

torch.autograd.set_detect_anomaly(True)

# +
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t in tensor:
            t.mul_(self.std[0]).add_(self.mean[0])
        return tensor
def get_device():
    if platform.platform().lower().startswith("mac"):  # macOS
        return "mps" if torch.backends.mps.is_available() else "cpu"
    else:  # Linux, Windows
        return "cuda" if torch.cuda.is_available() else "cpu"

class GANLoss(nn.Module):
    
    def __init__(self,real_label=1.0, fake_label = 0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.BCELoss()

    def get_target_label(self, prediction, is_real):
        if is_real:
            target = self.real_label
        else:
            target = self.fake_label
        return target.expand_as(prediction).type_as(prediction)

    def __call__(self, prediction, is_real):
        target = self.get_target_label(prediction, is_real)
        return self.loss(prediction, target)


# -

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # Output: (8, 14, 14)
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 7, 7)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 4, 4)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 2, 2)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=2, stride=1, padding=0),  # Output: (1, 1, 1)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.discriminator(x).view(-1, 1)

# +
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Unflatten(1, (64, 2, 2)),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=3, padding=1, output_padding=1),  # (32, 4, 4)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16, 8, 8)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (8, 16, 16)
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)

if __name__ == "__main__":
    input_tensor = torch.randn(8, 64*2*2) 
    generator = Generator()
    output = generator(input_tensor)
    print(output.shape)  # Should print torch.Size([8, 1, 28, 28])


# -

def get_device():
    if platform.platform().lower().startswith("mac"):  # macOS
        return "mps" if torch.backends.mps.is_available() else "cpu"
    else:  # Linux, Windows
        return "cuda" if torch.cuda.is_available() else "cpu"


# +
def plot(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.pause(0.001)
    # clear_output(wait=True)
    
def train(generator, discriminator, latent_dim, train_loader, optimizer, criterion, epochs, device='cpu'):
    generator.train()
    discriminator.train()
    # N = len(train_loader)
    M = len(train_loader.dataset)
    denorm = UnNormalize((0.5,), (0.5,))
    FIXED_NOISE = torch.randn((8, latent_dim)).to(device)
    
    for epoch in range(epochs):
        run_dloss, run_gloss = 0, 0
        pbar = tqdm(train_loader, desc=f'Epochs {epoch+1}/{epochs}', leave=True)
        for real_imgs, _ in pbar:

            real_imgs = real_imgs.to(device)
            
            # generate Noise
            z = torch.randn(real_imgs.shape[0], latent_dim).type_as(real_imgs)

            # Generator training
            fake_imgs = generator(z)
            
            y_hat_fake = discriminator(fake_imgs)
            g_loss = criterion(y_hat_fake, is_real=True)
            optimizer["gen"].zero_grad()
            g_loss.backward()
            optimizer["gen"].step()
            run_gloss += g_loss.item() * real_imgs.shape[0]

            # Discriminator training
            y_hat_real = discriminator(real_imgs)
            y_hat_fake = discriminator(fake_imgs.detach())
            real_loss = criterion(y_hat_real, is_real=True)
            fake_loss = criterion(y_hat_fake, is_real=False)
            d_loss = (real_loss + fake_loss) / 2

            optimizer["disc"].zero_grad()
            d_loss.backward()
            optimizer["disc"].step()
            run_dloss += d_loss.item() * real_imgs.shape[0]

            pbar.set_postfix(GL=f"{run_gloss / M:.3f}", DL=f"{run_dloss / M:.3f}")


        if (epoch + 1 )% 20 == 0 or (epoch==0):
            generator.eval()
            with torch.no_grad():
                fake_imgs = generator(FIXED_NOISE)
                fake_imgs = fake_imgs.view(-1, 1, 28, 28).cpu()
                fake_imgs = denorm(fake_imgs)
                img_grid_fake = torchvision.utils.make_grid(fake_imgs, nrow=8, normalize=True)
                plot(img_grid_fake.permute(1, 2, 0))
                generator.train()
            plt.show()
# -

if __name__ == "__main__":
    device = get_device()
    ROOT = "../DATASET/MNIST/test/"
    LR = 2e-4
    BATCH_SIZE = 1024
    Z_DIM = 64*2*2
    IMAGE_SIZE = 28 * 28
    EPOCHS = 200

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = MNIST(ROOT, train=False, transform=transform, download=False)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = {
        "disc":optim.Adam(discriminator.parameters(), lr=LR),
        "gen":optim.Adam(generator.parameters(), lr=LR)
    }

train(generator, discriminator, Z_DIM, dataloader, optimizer, GANLoss(), EPOCHS, device)


