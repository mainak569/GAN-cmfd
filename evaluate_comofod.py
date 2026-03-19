import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from dataset import CMFDDataset
from models import Generator
from metrics import evaluate_segmentation

# =====================
# Setup
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

test_img = "./data/test/images"
test_mask = "./data/test/masks"

test_ds = CMFDDataset(test_img, test_mask)
loader = DataLoader(test_ds, batch_size=1, shuffle=False)

print("Test samples:", len(test_ds))

# =====================
# Load Model
# =====================
G = Generator().to(device)

G.load_state_dict(
    torch.load("checkpoints/best_model.pth", map_location=device),
    strict=True
)

G.eval()

# =====================
# Collect Predictions
# =====================
gt_masks = []
pred_masks = []

with torch.no_grad():
    for img, mask in loader:

        img = img.to(device)
        mask = mask.to(device)

        output = G(img)

        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()

        # Remove batch + channel dims safely
        gt_np = mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

        gt_masks.append(gt_np)
        pred_masks.append(pred_np)

# =====================
# Evaluate
# =====================
results = evaluate_segmentation(gt_masks, pred_masks)

# =====================
# Print Results
# =====================
print("\n========== CoMoFoD TEST Evaluation ==========")

for k, v in results.items():
    print(f"{k}: {v:.4f}")

# =====================
# Save Results
# =====================
os.makedirs("results", exist_ok=True)

save_path = "results/comofod_test_results.json"

with open(save_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {save_path}")
