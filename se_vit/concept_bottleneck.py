"""
se_vit/concept_bottleneck.py
-----------------------------
Concept Bottleneck Layer (CBL): maps the [CLS] Transformer representation
to a vector of 512 clinically validated MRI biomarker probabilities.

Each dimension of the output corresponds to a named clinical concept
(e.g. hippocampal volume loss, entorhinal cortical thickness) defined
by consensus among board-certified neuroradiologists and neurologists.

Reference:
    Koh et al. (2020). Concept Bottleneck Models. ICML.
    Chen et al. (2025). SE-ViT. JMIAI 12(3), §4.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Concept registry — 512 clinical MRI biomarkers (abbreviated sample shown)
# Full registry loaded from configs/concepts.yaml at runtime
# ─────────────────────────────────────────────────────────────────────────────

CONCEPT_CATEGORIES = {
    "hippocampal":   list(range(0,   64)),   # 64 hippocampal features
    "entorhinal":    list(range(64,  128)),  # 64 entorhinal cortex features
    "ventricular":   list(range(128, 192)),  # 64 ventricular features
    "amygdala":      list(range(192, 256)),  # 64 amygdala features
    "temporal":      list(range(256, 320)),  # 64 temporal lobe features
    "frontal":       list(range(320, 384)),  # 64 frontal lobe features
    "cingulate":     list(range(384, 448)),  # 64 cingulate cortex features
    "white_matter":  list(range(448, 512)),  # 64 white matter features
}

IMPORTANCE_SCORES = {
    "hippocampal":  0.924,
    "entorhinal":   0.887,
    "ventricular":  0.861,
    "amygdala":     0.816,
    "temporal":     0.798,
    "cingulate":    0.754,
    "frontal":      0.743,
    "white_matter": 0.672,
}


class ConceptBottleneckLayer(nn.Module):
    """
    Maps Transformer [CLS] representation to 512 clinical concept probabilities.

    Architecture:
        z_L ∈ ℝ^{embed_dim}
          → Linear(embed_dim, hidden_dim) → GELU → Dropout
          → Linear(hidden_dim, num_concepts)
          → Sigmoid  →  c ∈ [0,1]^{num_concepts}

    The CBL is trained with binary cross-entropy on clinical concept
    annotations derived from AAL3 atlas-based automated segmentation
    (version 3.0) with expert validation (Cohen's κ = 0.81 ± 0.04).

    Args:
        embed_dim    (int): Input dimension from Transformer [CLS] token (768).
        num_concepts (int): Number of clinical concepts (512).
        hidden_dim   (int): Hidden layer size (default 1024).
        dropout      (float): Dropout rate (default 0.1).
    """

    def __init__(
        self,
        embed_dim:    int   = 768,
        num_concepts: int   = 512,
        hidden_dim:   int   = 1024,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.num_concepts = num_concepts

        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
        )

        # Learnable per-concept importance weights (initialised to 1)
        self.concept_importance = nn.Parameter(
            torch.ones(num_concepts), requires_grad=True)

        self._init_weights()

    def _init_weights(self):
        for module in self.bottleneck.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_repr: (B, embed_dim)  — [CLS] token from final Transformer layer.
        Returns:
            concept_probs: (B, num_concepts)  — concept probabilities in [0, 1].
        """
        logits        = self.bottleneck(cls_repr)              # (B, 512)
        concept_probs = torch.sigmoid(logits)                  # (B, 512)
        return concept_probs

    def concept_loss(
        self,
        concept_probs:  torch.Tensor,
        concept_labels: torch.Tensor,
        mask:           Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss over all 512 concept dimensions.

        Args:
            concept_probs  : (B, 512) predicted concept probabilities.
            concept_labels : (B, 512) binary ground-truth concept labels.
            mask           : (B, 512) optional mask for annotated concepts only.
        Returns:
            Scalar loss tensor.
        """
        loss = F.binary_cross_entropy(
            concept_probs, concept_labels.float(), reduction='none')   # (B, 512)

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def get_top_concepts(
        self,
        concept_probs:  torch.Tensor,
        concept_names:  Optional[List[str]] = None,
        top_k:          int = 20,
    ) -> List[Dict]:
        """
        Return top-k most activated concepts for each sample.

        Args:
            concept_probs : (B, 512) concept probabilities.
            concept_names : List of 512 concept name strings.
            top_k         : Number of top concepts to return.
        Returns:
            List of dicts (one per batch item), each mapping concept_name → score.
        """
        B = concept_probs.shape[0]
        results = []
        for b in range(B):
            scores, indices = concept_probs[b].topk(top_k)
            entry = {}
            for rank, (idx, score) in enumerate(
                    zip(indices.tolist(), scores.tolist())):
                name = concept_names[idx] if concept_names else f"concept_{idx}"
                entry[name] = round(score, 4)
            results.append(entry)
        return results

    def get_category_scores(self, concept_probs: torch.Tensor) -> Dict[str, float]:
        """
        Aggregate mean concept probability per anatomical category.

        Returns:
            Dict mapping category name → mean activation score.
        """
        scores = {}
        probs  = concept_probs.mean(dim=0)   # (512,) averaged over batch
        for category, indices in CONCEPT_CATEGORIES.items():
            scores[category] = probs[indices].mean().item()
        return scores

    def extra_repr(self) -> str:
        return (f"embed_dim={self.embed_dim}, "
                f"num_concepts={self.num_concepts}, "
                f"categories={list(CONCEPT_CATEGORIES.keys())}")
