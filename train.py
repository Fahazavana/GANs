import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter


device = 'mps' if torch.backends.mps.is_available() else 'cuda'

BATCH_SIZE = 64
LATENT_DIM = 100
N_EPOCHS = 500
LR_G = 2e-3
LR_D = 2e-3


transforms = Compose([ToTensor(), Normalize((0.5,), (0.5))])
mnist_full = MNIST(root='train', train=True, download=False, transform=transforms)

train = DataLoader(mnist_full, batch_size=BATCH_SIZE, shuffle=True)
generator = Generator(LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

opt_gen = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.5))
opt_dis = torch.optim.Adam(discriminator.parameters(), lr=LR_D)
criterion = nn.BCELoss()


VALIDATION  = torch.randn(64, LATENT_DIM).to(device)
history = {'g_loss':[], 'd_loss':[]}
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_dloss = SummaryWriter(f"runs/GAN_MNIST/gloss")
writer_gloss = SummaryWriter(f"runs/GAN_MNIST/dloss")
step = 0

for epoch in range(N_EPOCHS):
    # Progress bar
    N = len(train)
    pbar = tqdm(enumerate(train), total=N, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
    generator.train()
    discriminator.train()
    run_dloss = 0
    run_gloss = 0
    total = 0
    for k, (real_imgs, _) in pbar:
        real_imgs = real_imgs.to(device)
        # noise
        z = torch.randn(real_imgs.shape[0], LATENT_DIM).type_as(real_imgs)
        ########################################################################
        # generator  training max log (D(G(z)))
        
        fake_imgs = generator(z)
        y_hat = discriminator(fake_imgs)

        y_fake = torch.ones(real_imgs.shape[0], 1)
        y_fake = y_fake.type_as(real_imgs)
        g_loss = criterion(y_hat, y_fake)
        
        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()
        run_gloss += g_loss.item() 
        
        ########################################################################
        # discriminator training max log D(x) + log ( 1- D(G(z)))
        
        y_hat_real = discriminator(real_imgs)
        y_real = torch.ones(real_imgs.shape[0], 1).type_as(real_imgs)
        real_loss = criterion(y_hat_real, y_real)
        
        z = generator(z).detach()
        y_hat_fake = discriminator(z)
        y_fake = torch.zeros(real_imgs.shape[0], 1).type_as(real_imgs)
        fake_loss = criterion(y_hat_fake, y_fake)
        d_loss =  (real_loss + fake_loss)/2
        
        opt_dis.zero_grad()
        d_loss.backward()
        opt_dis.step()
        run_dloss += d_loss.item()
        
        # Update the progress bar
        pbar.set_postfix(g_loss = f"{run_gloss/N:.3f}", d_loss=f"{run_dloss/N:.3f}")
        
    with torch.no_grad():
        generator.eval()
        fake_imgs = generator(VALIDATION.type_as(real_imgs))
        fake_imgs = fake_imgs.view(-1, 1, 28, 28).cpu()
        img_grid_fake = torchvision.utils.make_grid(fake_imgs, normalize=True)
        writer_fake.add_image('Mnist fake', img_grid_fake , global_step=epoch+1)
    writer_gloss.add_scalar("Generator Loss", run_gloss/N, global_step=epoch+1)
    writer_dloss.add_scalar("Discriminator Loss", run_dloss/N, global_step=epoch+1)

    history['g_loss'].append(run_gloss)
    history['d_loss'].append(run_dloss)
    
    