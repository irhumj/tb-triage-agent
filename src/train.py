"""
train.py - fine-tune DenseNet121 to score chest X-rays for TB.

What this file does:
  1. Builds data loaders from the frozen splits (see data.py).
  2. Loads DenseNet121 pre-trained on ImageNet and replaces its last layer
     with a single output: the TB score.
  3. Trains with a class-weighted loss (TB is only ~9% of training images).
  4. Uses the val set ONLY for early stopping and picking the best epoch.
  5. Saves the best model and a training log, then reports internal-test AUROC.

Shenzhen and Montgomery are deliberately NOT touched here.
They are evaluated once, later, after calibration and thresholds are frozen.

Usage (on Kaggle):
    python train.py --splits /kaggle/input/tb-splits --out /kaggle/working
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.auto import tqdm

from data import make_datasets

SEED = 42


def set_seed(seed=SEED):
    """Same seed -> (nearly) the same results every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(device):
    """DenseNet121 from ImageNet, with 1000 classes replaced by 1 TB score."""
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model.to(device)


def make_loaders(ds, batch_size=32, workers=4):
    """Shuffle only the training data; the order of evaluation sets does not matter."""
    return {
        "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                            num_workers=workers, pin_memory=True),
        "val": DataLoader(ds["val"], batch_size=64, shuffle=False,
                          num_workers=workers, pin_memory=True),
        "internal_test": DataLoader(ds["internal_test"], batch_size=64, shuffle=False,
                                    num_workers=workers, pin_memory=True),
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One full pass over the training images. Returns the average loss."""
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()            # 1. forget the previous correction
        logits = model(x).squeeze(1)     # 2. the model guesses (raw scores)
        loss = criterion(logits, y)      # 3. how wrong was it? (TB mistakes weighted)
        loss.backward()                  # 4. work out which weights caused the error
        optimizer.step()                 # 5. adjust those weights a little
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    """Return raw scores (logits) and true labels for every image in a loader.

    Logits are kept (not probabilities) because calibration later needs them.
    """
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).squeeze(1).cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def evaluate(model, loader, device):
    """AUROC: how often a TB image is scored above a non-TB image."""
    logits, labels = predict(model, loader, device)
    return roc_auc_score(labels, logits)   # same ranking whether logits or probabilities


def main(splits_dir, out_dir, max_epochs=15, patience=4, lr=1e-4):
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    os.makedirs(out_dir, exist_ok=True)

    ds = make_datasets(splits_dir)
    loaders = make_loaders(ds)
    model = build_model(device)

    # Class weighting: each TB mistake counts (non-TB count / TB count) times more.
    train_labels = ds["train"].df["label"]
    pos_weight = torch.tensor(
        [(train_labels == 0).sum() / (train_labels == 1).sum()], device=device
    )
    print(f"pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_path = os.path.join(out_dir, "best_model.pt")
    best_auc, bad_epochs, history = 0.0, 0, []

    for epoch in range(1, max_epochs + 1):
        t = time.time()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_auc = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})
        print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} | "
              f"val AUROC {val_auc:.4f} | {time.time() - t:.0f}s")

        if val_auc > best_auc:          # better than ever -> keep this version
            best_auc, bad_epochs = val_auc, 0
            torch.save(model.state_dict(), best_path)
            print("   saved new best model")
        else:                           # no improvement -> count towards early stopping
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no improvement for {patience} epochs")
                break

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # Final check on the internal test set, using the best saved model.
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_auc = evaluate(model, loaders["internal_test"], device)
    print(f"\nBest val AUROC: {best_auc:.4f} | Internal test AUROC: {test_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TB screening model.")
    parser.add_argument("--splits", default="data/splits", help="folder with the split CSVs")
    parser.add_argument("--out", default="results", help="where to save model and log")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    main(args.splits, args.out, max_epochs=args.epochs, patience=args.patience)
