"""
se_vit/losses.py
-----------------
Composite loss function for SE-ViT training.

    L_total = L_CE(ŷ, y) + λ · L_concept(c, c*) + μ · L_cal

where:
    L_CE      = 4-class categorical cross-entropy on diagnostic labels
    L_concept = Binary cross-entropy over 512 concept annotations
    L_cal     = Temperature scaling calibration loss

Reference:
    Chen et al. (2025). SE-ViT. JMIAI 12(3), §4.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEViTLoss(nn.Module):
    """
    Composite SE-ViT loss combining diagnostic cross-entropy,
    concept annotation supervision, and calibration regularisation.

    Args:
        lambda_concept (float): Weight for concept loss (default 0.3).
        mu_cal         (float): Weight for calibration loss (default 0.05).
        num_classes    (int):   Number of diagnostic classes (default 4).
        label_smoothing(float): Label smoothing for cross-entropy (default 0.1).
    """

    def __init__(
        self,
        lambda_concept: float = 0.3,
        mu_cal:         float = 0.05,
        num_classes:    int   = 4,
        label_smoothing:float = 0.1,
    ):
        super().__init__()
        self.lambda_concept  = lambda_concept
        self.mu_cal          = mu_cal
        self.ce_loss         = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing)

    def forward(
        self,
        logits:         torch.Tensor,
        labels:         torch.Tensor,
        concept_probs:  torch.Tensor,
        concept_labels: Optional[torch.Tensor] = None,
        concept_mask:   Optional[torch.Tensor] = None,
        temperature:    float = 1.42,
    ) -> dict:
        """
        Compute composite loss.

        Args:
            logits         : (B, 4) raw diagnostic logits.
            labels         : (B,) integer diagnostic labels (0=CN,1=EMCI,2=LMCI,3=AD).
            concept_probs  : (B, 512) concept probabilities from CBL.
            concept_labels : (B, 512) binary concept annotations (optional).
            concept_mask   : (B, 512) mask for annotated concepts (optional).
            temperature    : Calibration temperature τ.

        Returns:
            Dict with keys: total, ce, concept, calibration.
        """
        # ── 1. Diagnostic cross-entropy
        l_ce = self.ce_loss(logits, labels)

        # ── 2. Concept annotation BCE (only when labels provided)
        if concept_labels is not None:
            l_concept = F.binary_cross_entropy(
                concept_probs,
                concept_labels.float(),
                reduction='none',
            )   # (B, 512)
            if concept_mask is not None:
                l_concept = (l_concept * concept_mask).sum() / (
                    concept_mask.sum() + 1e-8)
            else:
                l_concept = l_concept.mean()
        else:
            l_concept = torch.tensor(0.0, device=logits.device)

        # ── 3. Calibration loss: NLL of temperature-scaled predictions
        cal_logits = logits / temperature
        l_cal      = F.cross_entropy(cal_logits, labels)

        # ── Total
        total = l_ce + self.lambda_concept * l_concept + self.mu_cal * l_cal

        return {
            "total":       total,
            "ce":          l_ce,
            "concept":     l_concept,
            "calibration": l_cal,
        }


class FocalLoss(nn.Module):
    """
    Optional focal loss for class-imbalanced ADNI batches.
    Reduces loss contribution of well-classified easy examples.

    Args:
        gamma (float): Focusing parameter (default 2.0).
        alpha (float): Optional class weight scalar.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        ce     = F.cross_entropy(logits, labels, reduction='none')
        pt     = torch.exp(-ce)
        loss   = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha * loss
        return loss.mean()


class TemperatureScaling(nn.Module):
    """
    Post-hoc temperature scaling calibration.
    Fits a single scalar τ on the validation set to minimise NLL.

    Usage:
        calibrator = TemperatureScaling()
        calibrator.fit(val_logits, val_labels)
        cal_probs = calibrator(test_logits)

    Reference:
        Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=-1)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Optimise temperature using LBFGS on validation logits."""
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        def eval_fn():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        tau = self.temperature.item()
        print(f"[TemperatureScaling] Optimal τ = {tau:.4f}")
        return tau


def expected_calibration_error(
    probs:  torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width bins.

    Args:
        probs  : (N, C) softmax probabilities.
        labels : (N,) integer class labels.
        n_bins : Number of confidence bins (default 10).

    Returns:
        ECE as a float.
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece     = 0.0
    bins    = torch.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean().item()
        bin_acc  = accuracies[mask].float().mean().item()
        bin_size = mask.sum().item() / len(labels)
        ece     += abs(bin_acc - bin_conf) * bin_size

    return ece
