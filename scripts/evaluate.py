"""
scripts/evaluate.py
--------------------
Evaluation script: generates per-class metrics, confusion matrix,
concept attributions, and explanation maps for the SE-ViT model.

Usage:
    python scripts/evaluate.py \\
        --config     configs/sevit_base.yaml \\
        --checkpoint outputs/sevit_run1/best_model.pth \\
        --data_dir   data/processed \\
        --output_dir outputs/explanations
"""

import sys, yaml, argparse, json
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from se_vit import SEViT, expected_calibration_error
from scripts.train import ADNIDataset

CLASS_NAMES = ["CN", "EMCI", "LMCI", "AD"]


def compute_metrics(probs, labels):
    """Compute accuracy, per-class sensitivity/specificity/F1, macro AUC."""
    preds    = probs.argmax(dim=1)
    accuracy = preds.eq(labels).float().mean().item()

    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = ((preds == c) & (labels == c)).sum().float()
        fp = ((preds == c) & (labels != c)).sum().float()
        tn = ((preds != c) & (labels != c)).sum().float()
        fn = ((preds != c) & (labels == c)).sum().float()
        sens = (tp / (tp + fn + 1e-8)).item()
        spec = (tn / (tn + fp + 1e-8)).item()
        prec = (tp / (tp + fp + 1e-8)).item()
        f1   = 2 * prec * sens / (prec + sens + 1e-8)
        per_class[name] = dict(sensitivity=round(sens,4),
                               specificity=round(spec,4),
                               f1=round(f1,4))

    # Macro AUC via one-vs-rest trapezoidal rule
    auc_list = []
    for c in range(len(CLASS_NAMES)):
        scores = probs[:, c].numpy()
        binary = (labels.numpy() == c).astype(int)
        order  = np.argsort(-scores)
        tpr, fpr, tp_cum, fp_cum = [0.0], [0.0], 0, 0
        n_pos = binary.sum(); n_neg = len(binary) - n_pos
        for idx in order:
            if binary[idx]:
                tp_cum += 1
            else:
                fp_cum += 1
            tpr.append(tp_cum / (n_pos + 1e-8))
            fpr.append(fp_cum / (n_neg + 1e-8))
        auc = float(np.trapz(tpr, fpr))
        auc_list.append(abs(auc))
    macro_auc = float(np.mean(auc_list))

    return accuracy, per_class, macro_auc


def confusion_matrix(preds, labels, n_classes=4):
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for p, l in zip(preds, labels):
        cm[l.item(), p.item()] += 1
    return cm


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model
    model_cfg = cfg.get("model", {})
    model = SEViT(
        img_size     = model_cfg.get("img_size",     224),
        patch_size   = model_cfg.get("patch_size",   16),
        embed_dim    = model_cfg.get("embed_dim",    768),
        depth        = model_cfg.get("depth",        12),
        num_heads    = model_cfg.get("num_heads",    12),
        num_classes  = model_cfg.get("num_classes",  4),
        num_concepts = model_cfg.get("num_concepts", 512),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[Evaluate] Loaded checkpoint from {args.checkpoint}")

    # ── Test dataloader
    test_ds     = ADNIDataset(args.data_dir, split="test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    all_probs, all_labels, all_concepts = [], [], []
    explanation_samples = []

    print(f"[Evaluate] Running inference on {len(test_ds)} test slices...")
    with torch.no_grad():
        for batch_idx, (x, labels, concept_labels) in enumerate(test_loader):
            x      = x.to(device)
            output = model(x, return_explanations=(batch_idx < 5))

            all_probs.append(output.probabilities.cpu())
            all_labels.append(labels)
            all_concepts.append(output.concept_probs.cpu())

            if batch_idx < 5 and output.concept_attributions is not None:
                explanation_samples.append({
                    "batch":        batch_idx,
                    "predictions":  output.probabilities.argmax(1).tolist(),
                    "true_labels":  labels.tolist(),
                    "uncertainty":  output.uncertainty.cpu().tolist(),
                })

    all_probs   = torch.cat(all_probs)
    all_labels  = torch.cat(all_labels)
    all_concepts= torch.cat(all_concepts)
    all_preds   = all_probs.argmax(dim=1)

    # ── Metrics
    accuracy, per_class, macro_auc = compute_metrics(all_probs, all_labels)
    ece = expected_calibration_error(all_probs, all_labels)
    cm  = confusion_matrix(all_preds, all_labels)

    print(f"\n{'='*55}")
    print(f"TEST SET RESULTS  (n={len(all_labels)})")
    print(f"{'='*55}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  AUC-ROC   : {macro_auc:.4f}")
    print(f"  ECE       : {ece:.4f}")
    print(f"\nPer-class metrics:")
    for cls, m in per_class.items():
        print(f"  {cls:5s}: Sens={m['sensitivity']:.3f}  "
              f"Spec={m['specificity']:.3f}  F1={m['f1']:.3f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    header = "       " + "  ".join(f"{c:6s}" for c in CLASS_NAMES)
    print(header)
    for i, row_name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i,j].item():6d}" for j in range(4))
        print(f"  {row_name:5s}: {row}")
    print(f"{'='*55}")

    # ── Mean concept importance per category
    mean_concepts = all_concepts.mean(dim=0)   # (512,)
    concept_cats  = {
        "hippocampal":  mean_concepts[0:64].mean().item(),
        "entorhinal":   mean_concepts[64:128].mean().item(),
        "ventricular":  mean_concepts[128:192].mean().item(),
        "amygdala":     mean_concepts[192:256].mean().item(),
        "temporal":     mean_concepts[256:320].mean().item(),
        "frontal":      mean_concepts[320:384].mean().item(),
        "cingulate":    mean_concepts[384:448].mean().item(),
        "white_matter": mean_concepts[448:512].mean().item(),
    }
    print("\nMean concept activation by anatomical category:")
    for cat, score in sorted(concept_cats.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 30)
        print(f"  {cat:15s}: {score:.3f} {bar}")

    # ── Save results
    results = {
        "accuracy":         round(accuracy, 4),
        "auc_roc":          round(macro_auc, 4),
        "ece":              round(ece, 4),
        "per_class":        per_class,
        "concept_categories": {k: round(v, 4) for k, v in concept_cats.items()},
        "confusion_matrix": cm.tolist(),
        "n_test":           len(all_labels),
        "explanation_samples": explanation_samples,
    }
    out_path = out_dir / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Results → {out_path}")

    # ── High-uncertainty cases (flag for specialist review)
    model.eval()
    uncertain_cases = []
    with torch.no_grad():
        for x, labels, _ in test_loader:
            output = model(x.to(device))
            unc    = output.uncertainty.cpu()
            high   = unc > unc.quantile(0.75)
            if high.any():
                for i, flag in enumerate(high):
                    if flag:
                        uncertain_cases.append({
                            "true_label": CLASS_NAMES[labels[i].item()],
                            "pred_label": CLASS_NAMES[output.probabilities[i].argmax().item()],
                            "uncertainty": round(unc[i].item(), 4),
                        })
    pct_flagged = len(uncertain_cases) / len(all_labels) * 100
    print(f"\n[Uncertainty] {len(uncertain_cases)} cases flagged for specialist review "
          f"({pct_flagged:.1f}% — paper reports 8.3%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SE-ViT and generate explanations")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--output_dir",  required=True)
    main(parser.parse_args())
