"""
scripts/train.py
-----------------
Training script for SE-ViT on ADNI MRI data.

Usage:
    python scripts/train.py \\
        --config configs/sevit_base.yaml \\
        --data_dir data/processed \\
        --output_dir outputs/sevit_run1 \\
        --seed 42

Reproduces results from:
    Chen et al. (2025). SE-ViT. JMIAI 12(3), §4.5.
    Expected: Accuracy 91.2 ± 0.9%, AUC-ROC 0.954
"""

import os
import sys
import yaml
import argparse
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# SE-ViT imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from se_vit import SEViT, SEViTLoss
from se_vit.losses import TemperatureScaling, expected_calibration_error


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Seed all random number generators for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Seed] All RNGs seeded with seed={seed}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (ADNI MRI)
# ─────────────────────────────────────────────────────────────────────────────

class ADNIDataset(torch.utils.data.Dataset):
    """
    ADNI T1-weighted MRI dataset.
    Expects preprocessed 224×224 axial slices saved as .pt tensors.

    Directory layout:
        data_dir/
          train/ CN/ EMCI/ LMCI/ AD/  (each contains *.pt slice tensors)
          val/   CN/ EMCI/ LMCI/ AD/
          test/  CN/ EMCI/ LMCI/ AD/

    Each .pt file is a (1, 224, 224) float32 tensor (single MRI slice).
    """

    CLASS_MAP = {"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}

    def __init__(self, data_dir: str, split: str = "train",
                 augment: bool = False):
        self.data_dir = Path(data_dir) / split
        self.augment  = augment
        self.samples  = []   # list of (path, label, concept_label_path)

        for class_name, label in self.CLASS_MAP.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for pt_file in sorted(class_dir.glob("*.pt")):
                concept_path = pt_file.with_suffix('.concepts.pt')
                self.samples.append((pt_file, label,
                                     concept_path if concept_path.exists() else None))

        print(f"[ADNIDataset] {split}: {len(self.samples)} slices loaded "
              f"from {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, label, concept_path = self.samples[idx]
        x = torch.load(pt_path, weights_only=True)   # (1, 224, 224)

        if self.augment:
            x = self._augment(x)

        concept_labels = (torch.load(concept_path, weights_only=True)
                          if concept_path else torch.zeros(512))

        return x, torch.tensor(label, dtype=torch.long), concept_labels

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """In-training augmentation pipeline."""
        import torchvision.transforms.functional as TF
        # Random horizontal flip
        if random.random() > 0.5:
            x = TF.hflip(x)
        # Random rotation ±10°
        angle = random.uniform(-10, 10)
        x = TF.rotate(x, angle)
        # Random Gaussian noise
        sigma = random.uniform(0.01, 0.05)
        x = x + torch.randn_like(x) * sigma
        return x.clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = correct = total = 0

    for batch_idx, (x, labels, concept_labels) in enumerate(loader):
        x              = x.to(device)
        labels         = labels.to(device)
        concept_labels = concept_labels.to(device)

        optimizer.zero_grad()
        with autocast():
            output = model(x, return_explanations=False,
                           concept_labels=concept_labels)
            losses = criterion(
                logits=output.logits,
                labels=labels,
                concept_probs=output.concept_probs,
                concept_labels=concept_labels,
            )

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses["total"].item()
        preds       = output.logits.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += labels.size(0)

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}]  "
                  f"loss={losses['total']:.4f}  "
                  f"ce={losses['ce']:.4f}  "
                  f"concept={losses['concept']:.4f}")

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_probs  = []
    all_labels = []

    for x, labels, concept_labels in loader:
        x              = x.to(device)
        labels         = labels.to(device)
        concept_labels = concept_labels.to(device)

        output = model(x)
        losses = criterion(
            logits=output.logits,
            labels=labels,
            concept_probs=output.concept_probs,
            concept_labels=concept_labels,
        )

        total_loss += losses["total"].item()
        preds       = output.logits.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += labels.size(0)
        all_probs.append(output.probabilities.cpu())
        all_labels.append(labels.cpu())

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    ece        = expected_calibration_error(all_probs, all_labels)

    return total_loss / len(loader), correct / total, ece


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed or cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device} | PyTorch {torch.__version__}")

    # ── Datasets & Loaders
    train_ds = ADNIDataset(args.data_dir, split="train",  augment=True)
    val_ds   = ADNIDataset(args.data_dir, split="val",    augment=False)
    test_ds  = ADNIDataset(args.data_dir, split="test",   augment=False)

    train_cfg = cfg.get("training", {})
    bs = train_cfg.get("batch_size", 32)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model
    model_cfg = cfg.get("model", {})
    model = SEViT(
        img_size     = model_cfg.get("img_size",     224),
        patch_size   = model_cfg.get("patch_size",   16),
        embed_dim    = model_cfg.get("embed_dim",    768),
        depth        = model_cfg.get("depth",        12),
        num_heads    = model_cfg.get("num_heads",    12),
        num_classes  = model_cfg.get("num_classes",  4),
        num_concepts = model_cfg.get("num_concepts", 512),
        drop_rate    = model_cfg.get("dropout",      0.1),
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"[GPU] Using {torch.cuda.device_count()} GPUs")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {n_params/1e6:.1f}M")

    # ── Loss, optimiser, scheduler
    criterion = SEViTLoss(
        lambda_concept=train_cfg.get("loss_weights", {}).get("concept", 0.3),
        mu_cal        =train_cfg.get("loss_weights", {}).get("calibration", 0.05),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = float(train_cfg.get("lr", 1e-4)),
        weight_decay = float(train_cfg.get("weight_decay", 0.01)),
    )

    epochs   = train_cfg.get("epochs", 100)
    warmup   = train_cfg.get("warmup_epochs", 10)

    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(enabled=train_cfg.get("mixed_precision", True))

    # ── Training loop
    best_val_auc = 0.0
    best_epoch   = 0
    history      = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "val_ece":[]}

    print(f"\n{'='*60}")
    print(f"Training SE-ViT for {epochs} epochs")
    print(f"Seed: {seed} | Batch size: {bs} | LR: {train_cfg.get('lr', 1e-4)}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch)
        vl_loss, vl_acc, vl_ece = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_ece"].append(vl_ece)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f} | "
              f"ECE={vl_ece:.4f} | lr={lr_now:.2e}")

        # Save best model by validation accuracy
        if vl_acc > best_val_auc:
            best_val_auc = vl_acc
            best_epoch   = epoch
            ckpt_path    = out_dir / "best_model.pth"
            state = model.module.state_dict() if hasattr(model, 'module') \
                    else model.state_dict()
            torch.save({
                "epoch":            epoch,
                "model_state_dict": state,
                "val_acc":          vl_acc,
                "val_ece":          vl_ece,
                "config":           cfg,
                "seed":             seed,
            }, ckpt_path)
            print(f"  ✅ Best model saved (epoch {epoch}, val_acc={vl_acc:.4f})")

    # ── Temperature calibration on validation set
    print("\n[Calibration] Fitting temperature scaling on validation set...")
    calibrator = TemperatureScaling().to(device)
    best_model = SEViT(**{k: model_cfg.get(k, v) for k, v in {
        'img_size':224,'patch_size':16,'embed_dim':768,'depth':12,
        'num_heads':12,'num_classes':4,'num_concepts':512,'drop_rate':0.1
    }.items()}).to(device)
    ckpt = torch.load(out_dir / "best_model.pth", map_location=device)
    best_model.load_state_dict(ckpt["model_state_dict"])
    best_model.eval()
    val_logits, val_labels_all = [], []
    with torch.no_grad():
        for x, labels, _ in val_loader:
            val_logits.append(best_model(x.to(device)).logits)
            val_labels_all.append(labels.to(device))
    tau = calibrator.fit(torch.cat(val_logits), torch.cat(val_labels_all))
    print(f"[Calibration] τ = {tau:.4f} (paper reports τ = 1.42)")

    # ── Final test evaluation
    print("\n[Test] Evaluating on held-out test set...")
    te_loss, te_acc, te_ece = evaluate(best_model, test_loader, criterion, device)
    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS (best model from epoch {best_epoch})")
    print(f"  Test Accuracy : {te_acc:.4f}  (paper: 0.912)")
    print(f"  Test ECE      : {te_ece:.4f}  (paper: 0.034)")
    print(f"{'='*60}")

    # Save training history
    import json
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Done] Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SE-ViT on ADNI MRI")
    parser.add_argument("--config",     required=True,  help="Path to YAML config")
    parser.add_argument("--data_dir",   required=True,  help="Preprocessed ADNI data dir")
    parser.add_argument("--output_dir", required=True,  help="Output directory")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
