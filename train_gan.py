import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CMFDDataset
from models import Generator, Discriminator, weights_init
import os

# =====================
# Settings
# =====================
train_img = "./data/train/images"
train_mask = "./data/train/masks"
val_img = "./data/val/images"
val_mask = "./data/val/masks"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

batch_size = 8        # DGX can handle this
epochs = 100
lambda_dice = 10

# =====================
# Dice Loss (important)
# =====================
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# =====================
# Data
# =====================
train_ds = CMFDDataset(train_img, train_mask)
val_ds = CMFDDataset(val_img, val_mask)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =====================
# Models
# =====================
G = Generator().to(device)
D = Discriminator().to(device)

G.apply(weights_init)
D.apply(weights_init)

opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

bce = nn.BCEWithLogitsLoss()  # GAN loss

# IMPORTANT: handle class imbalance
pos_weight = torch.tensor([5.0]).to(device)
bce_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

os.makedirs("checkpoints", exist_ok=True)

best_val = 1e10

# =====================
# Training Loop
# =====================
for epoch in range(epochs):

    G.train()
    D.train()

    running_G = 0
    running_D = 0

    for img, mask in train_loader:

        img = img.to(device)
        mask = mask.to(device)

        # -----------------
        # Train Discriminator
        # -----------------
        fake_mask = G(img)

        D_real = D(img, mask)
        D_fake = D(img, fake_mask.detach())

        loss_D_real = bce(D_real, torch.ones_like(D_real))
        loss_D_fake = bce(D_fake, torch.zeros_like(D_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # -----------------
        # Train Generator
        # -----------------
        fake_mask = G(img)
        D_fake = D(img, fake_mask)

        loss_G_adv = bce(D_fake, torch.ones_like(D_fake))

        loss_G_bce = bce_seg(fake_mask, mask) * 20
        loss_G_dice = dice_loss(fake_mask, mask) * 10

        # Slightly reduce GAN weight for stability
        loss_G = 0.5 * loss_G_adv + loss_G_bce + loss_G_dice

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        running_G += loss_G.item()
        running_D += loss_D.item()

    # =====================
    # Validation
    # =====================
    G.eval()
    val_loss = 0

    with torch.no_grad():
        for img, mask in val_loader:
            img = img.to(device)
            mask = mask.to(device)

            output = G(img)
            val_loss += (
                bce_seg(output, mask) +
                dice_loss(output, mask)
            ).item()

    avg_G = running_G / len(train_loader)
    avg_D = running_D / len(train_loader)
    avg_val = val_loss / len(val_loader)

    print(f"Epoch {epoch:03d} | D: {avg_D:.3f} | G: {avg_G:.3f} | Val Loss: {avg_val:.4f}")

    # Save best model
    if avg_val < best_val:
        best_val = avg_val
        torch.save(G.state_dict(), "checkpoints/best_model.pth")

    # Save periodic checkpoint
    if epoch % 20 == 0:
        torch.save(G.state_dict(), f"checkpoints/G_epoch_{epoch}.pth")

print("Training complete")
