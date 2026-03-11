"""
STEP 4 — Train TEDNet 2RX
==========================
CNN Encoder + Transformer to predict 17 COCO keypoints from CSI windows.

Input  : X.npy  (N, window_size, 256)   — CSI from 2 receivers
Output : Y.npy  (N, 34)                  — normalised x,y keypoints

Architecture (TEDNet 2RX):
  CNN Encoder  → local temporal-subcarrier features
  Positional   → learnable position embeddings
  Transformer  → global temporal attention
  Regressor    → FC head → 34 values (17 joints × x,y)
"""

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import json

# ===================== CONFIGURATION =====================
DATA_DIR    = os.path.join("data", "processed")
MODEL_DIR   = "models"
LOG_PATH    = os.path.join(MODEL_DIR, "train_log.json")

WINDOW_SIZE = 100
N_FEATURES  = 256     # 128 sub × 2 receivers
N_JOINTS    = 17
OUTPUT_DIM  = N_JOINTS * 2   # x, y only

# Training hyper-parameters
BATCH_SIZE  = 64
NUM_EPOCHS  = 100
LR          = 3e-4
WEIGHT_DECAY= 1e-4
PATIENCE    = 15      # early stopping

# Transformer hyper-parameters
D_MODEL     = 128
N_HEADS     = 8
N_LAYERS    = 4
DIM_FF      = 512
DROPOUT     = 0.1

os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ===================== DATASET =====================

class CSIDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ===================== MODEL =====================

class CNNEncoder(nn.Module):
    """1D CNN across time to extract local features."""
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, d_model,     kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model,     d_model,     kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model,     d_model * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model,     kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, T, C) → permute → (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)   # back to (B, T, d_model)
        return x


class TEDNet(nn.Module):
    """
    TEDNet 2RX:
      CNN Encoder → Transformer Encoder → FC Regressor
    """
    def __init__(self, window_size=WINDOW_SIZE, n_features=N_FEATURES,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 dim_ff=DIM_FF, dropout=DROPOUT, output_dim=OUTPUT_DIM):
        super().__init__()

        self.cnn_encoder = CNNEncoder(n_features, d_model)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Regression head
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
            nn.Sigmoid(),    # normalise output to [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x):
        # x: (B, T, C)
        x = self.cnn_encoder(x)       # (B, T, d_model)
        x = x + self.pos_embed        # add positional embedding
        x = self.transformer(x)       # (B, T, d_model)
        x = x.mean(dim=1)             # global average pooling over time
        x = self.regressor(x)         # (B, output_dim)
        return x


# ===================== LOSS =====================

class WingLoss(nn.Module):
    """Wing loss — better for small keypoint errors."""
    def __init__(self, w=10.0, epsilon=2.0):
        super().__init__()
        self.w = w
        self.e = epsilon
        self.C = w - w * np.log(1 + w / epsilon)

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.e),
            diff - self.C
        )
        return loss.mean()


# ===================== TRAINING LOOP =====================

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimiser.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            pred    = model(X_batch)
            loss    = criterion(pred, Y_batch)
            total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def mpjpe(pred, target):
    """Mean Per Joint Position Error (pixels in normalised [0,1] space)."""
    pred_j   = pred.reshape(-1, 17, 2)
    target_j = target.reshape(-1, 17, 2)
    return torch.sqrt(((pred_j - target_j) ** 2).sum(dim=-1)).mean().item()


# ===================== MAIN =====================

if __name__ == "__main__":
    print("=" * 55)
    print("WiFi-Pose — Training TEDNet 2RX")
    print("=" * 55)

    # ── Load data ──────────────────────────────────────────────
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y.npy"))
    print(f"X: {X.shape}  Y: {Y.shape}")

    dataset = CSIDataset(X, Y)
    n_val   = max(1, int(0.15 * len(dataset)))
    n_test  = max(1, int(0.10 * len(dataset)))
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")

    # ── Model ──────────────────────────────────────────────────
    model     = TEDNet().to(DEVICE)
    criterion = WingLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TEDNet parameters: {n_params:,}")

    # ── Train ──────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_ctr  = 0
    log = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_dl, optimiser, criterion, DEVICE)
        val_loss   = val_epoch(model, val_dl, criterion, DEVICE)
        scheduler.step()

        # Compute MPJPE on validation set
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for Xb, Yb in val_dl:
                preds.append(model(Xb.to(DEVICE)).cpu())
                targets.append(Yb)
        preds   = torch.cat(preds)
        targets = torch.cat(targets)
        err     = mpjpe(preds, targets)

        log.append({"epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "mpjpe": err})

        print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}]  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  MPJPE: {err:.4f}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "tednet_best.pth"))
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ── Test Evaluation ────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "tednet_best.pth")))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for Xb, Yb in test_dl:
            preds.append(model(Xb.to(DEVICE)).cpu())
            targets.append(Yb)
    preds   = torch.cat(preds)
    targets = torch.cat(targets)
    test_err = mpjpe(preds, targets)

    print(f"\n✅ Best val loss : {best_val_loss:.4f}")
    print(f"✅ Test MPJPE    : {test_err:.4f}  (normalised [0,1] space)")

    # Save log
    with open(LOG_PATH, "w") as f:
        json.dump({"log": log, "test_mpjpe": test_err}, f, indent=2)
    print(f"✅ Log saved → {LOG_PATH}")
    print(f"✅ Model saved → {MODEL_DIR}/tednet_best.pth")