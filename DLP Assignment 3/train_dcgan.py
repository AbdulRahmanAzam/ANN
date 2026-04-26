"""
DCGAN training script for Assignment 3.
Saves all outputs to samples/ and prints progress.
"""
import matplotlib
matplotlib.use('Agg')   # no display needed

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ── Setup ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS     = 50
LR         = 0.0002
BETA1      = 0.5
SEED       = 42
torch.manual_seed(SEED)

os.makedirs('samples', exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f'Dataset: {len(dataset)} images  |  {len(dataloader)} batches/epoch')

# ── Models ────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc  = nn.Linear(latent_dim, 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(self.fc(z).view(-1, 128, 7, 7))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64,  kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(self.net(x).view(-1, 128 * 7 * 7)))

def weights_init(m):
    cls = m.__class__.__name__
    if 'Conv' in cls:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in cls:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)
G.apply(weights_init); D.apply(weights_init)

criterion = nn.BCELoss()
G_optim = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
D_optim = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
fixed_noise = torch.randn(16, LATENT_DIM, device=device)

# ── Training ──────────────────────────────────────────────────────────────────
d_losses, g_losses = [], []

for epoch in range(EPOCHS):
    d_run, g_run = 0.0, 0.0

    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)
        bs = real_imgs.size(0)
        real_labels = torch.ones(bs, 1, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)

        # Discriminator step
        D_optim.zero_grad()
        loss_real = criterion(D(real_imgs), real_labels)
        z = torch.randn(bs, LATENT_DIM, device=device)
        fake_imgs = G(z)
        loss_fake = criterion(D(fake_imgs.detach()), fake_labels)
        d_loss = loss_real + loss_fake
        d_loss.backward(); D_optim.step()

        # Generator step
        G_optim.zero_grad()
        g_loss = criterion(D(fake_imgs), real_labels)
        g_loss.backward(); G_optim.step()

        d_run += d_loss.item()
        g_run += g_loss.item()

    avg_d = d_run / len(dataloader)
    avg_g = g_run / len(dataloader)
    d_losses.append(avg_d)
    g_losses.append(avg_g)
    print(f'Epoch [{epoch+1:>2}/{EPOCHS}]  D: {avg_d:.4f}  G: {avg_g:.4f}', flush=True)

    if (epoch + 1) % 10 == 0:
        G.eval()
        with torch.no_grad():
            imgs = G(fixed_noise).cpu()
        G.train()
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for ax, img in zip(axes.flat, imgs):
            ax.imshow(img.squeeze(), cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
        plt.suptitle(f'Generated — Epoch {epoch+1}', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'samples/epoch_{epoch+1:02d}.png', bbox_inches='tight')
        plt.close()
        print(f'  Saved samples/epoch_{epoch+1:02d}.png', flush=True)

# ── Final outputs ─────────────────────────────────────────────────────────────
# Loss curve
plt.figure(figsize=(8, 4))
plt.plot(d_losses, label='Discriminator')
plt.plot(g_losses, label='Generator')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('DCGAN Training Losses')
plt.legend(); plt.tight_layout()
plt.savefig('samples/loss_curve.png')
plt.close()

# 25 generated digits
G.eval()
with torch.no_grad():
    final_imgs = G(torch.randn(25, LATENT_DIM, device=device)).cpu()
fig, axes = plt.subplots(5, 5, figsize=(7, 7))
for ax, img in zip(axes.flat, final_imgs):
    ax.imshow(img.squeeze(), cmap='gray', vmin=-1, vmax=1)
    ax.axis('off')
plt.suptitle('25 Generated Handwritten Digits (Final Model)', fontsize=12)
plt.tight_layout()
plt.savefig('samples/final_generated.png', bbox_inches='tight')
plt.close()

# Save weights
torch.save(G.state_dict(), 'samples/generator.pth')
torch.save(D.state_dict(), 'samples/discriminator.pth')
print('Done. Files in samples/:', os.listdir('samples'))
