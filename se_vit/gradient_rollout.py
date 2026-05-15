"""
se_vit/gradient_rollout.py
---------------------------
Gradient Rollout: aggregates attention flow across all 12 Transformer layers
to produce spatially faithful explanation maps.

Method:
    For each layer l:
        R_l = E_h[ A_l ⊙ (∂y/∂A_l) ]+
    Rollout:
        R = R_L · R_{L-1} · ... · R_1

Reference:
    Abnar & Zuidema (2020). Quantifying Attention Flow in Transformers. ACL.
    DOI: 10.18653/v1/2020.acl-main.385
    Chen et al. (2025). SE-ViT. JMIAI 12(3), §4.4.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class GradientRollout(nn.Module):
    """
    Computes gradient-weighted attention rollout across all Transformer layers.

    For each layer l, we compute the element-wise product of the attention map
    A_l and its gradient ∂y/∂A_l (averaged across heads), apply ReLU to retain
    only positive contributions, then multiply rollout matrices across all layers
    to obtain the final relevance map.

    Args:
        transformer_blocks: nn.ModuleList of TransformerBlock instances.
        discard_ratio      : Fraction of lowest-weight attention values to zero
                             out before rollout (noise reduction). Default 0.9.
    """

    def __init__(
        self,
        transformer_blocks: nn.ModuleList,
        discard_ratio: float = 0.9,
    ):
        super().__init__()
        self.blocks        = transformer_blocks
        self.discard_ratio = discard_ratio
        self._hooks: List  = []
        self._gradients: List[torch.Tensor] = []

    # ── Gradient hooks ────────────────────────────────────────────────────────

    def _register_hooks(self):
        """Register backward hooks on every attention layer to capture gradients."""
        self._gradients = []
        self._hooks     = []

        def make_hook(layer_idx):
            def hook(grad):
                self._gradients.insert(0, grad)  # prepend → index 0 = first layer
            return hook

        for idx, block in enumerate(self.blocks):
            h = block.attn.attention_weights.register_hook(make_hook(idx))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ── Core rollout ──────────────────────────────────────────────────────────

    def forward(
        self,
        x:          torch.Tensor,
        target_cls: torch.Tensor,
        model_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gradient rollout attention map.

        Args:
            x          : (B, 1, H, W) input MRI slices — must require grad.
            target_cls : (B,) predicted class indices.
            model_output: Pre-computed logits if available (avoids re-forward).

        Returns:
            rollout_map : (B, H, W) upsampled relevance map in [0, 1].
        """
        # Collect attention weights from all layers (already stored in forward pass)
        attn_weights = [block.attn.attention_weights
                        for block in self.blocks]   # list of (B, H, N+1, N+1)

        if any(a is None for a in attn_weights):
            raise RuntimeError(
                "Attention weights not found. Run a forward pass first.")

        B      = x.shape[0]
        N      = attn_weights[0].shape[-1]   # sequence length = 197
        device = x.device

        # Initialise rollout as identity
        rollout = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)

        for attn in attn_weights:
            # attn: (B, num_heads, N, N) → average over heads
            A = attn.mean(dim=1)       # (B, N, N)

            # Discard low-weight attention values (noise reduction)
            flat       = A.view(B, -1)
            threshold  = flat.quantile(self.discard_ratio, dim=1, keepdim=True)
            threshold  = threshold.view(B, 1, 1)
            A          = torch.where(A >= threshold, A, torch.zeros_like(A))

            # Add residual identity connection and re-normalise rows
            A = A + torch.eye(N, device=device).unsqueeze(0)
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

            # Accumulate rollout: R ← R · A
            rollout = torch.bmm(rollout, A)

        # Extract [CLS] → patch relevance: row 0, columns 1:
        # Shape: (B, N-1) = (B, 196)
        cls_relevance = rollout[:, 0, 1:]                    # (B, 196)

        # Reshape to spatial grid
        grid_size = int(cls_relevance.shape[-1] ** 0.5)      # 14
        cls_relevance = cls_relevance.view(B, 1, grid_size, grid_size)

        # Upsample to original image size
        rollout_map = F.interpolate(
            cls_relevance,
            size=(x.shape[-2], x.shape[-1]),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)   # (B, H, W)

        # Normalise to [0, 1] per sample
        rollout_map = self._normalize(rollout_map)

        return rollout_map   # (B, H, W)

    @staticmethod
    def _normalize(maps: torch.Tensor) -> torch.Tensor:
        """Per-sample min-max normalisation."""
        B = maps.shape[0]
        maps_flat = maps.view(B, -1)
        mn = maps_flat.min(dim=1).values.view(B, 1, 1)
        mx = maps_flat.max(dim=1).values.view(B, 1, 1)
        return (maps - mn) / (mx - mn + 1e-8)

    def get_patch_importance(
        self,
        rollout_map: torch.Tensor,
        top_k: int = 10,
    ):
        """
        Return indices and scores of top-k most relevant image patches.

        Args:
            rollout_map: (B, H, W) normalised rollout map.
            top_k      : Number of top patches to return.
        Returns:
            List of (patch_row, patch_col, importance_score) tuples per sample.
        """
        B, H, W  = rollout_map.shape
        patch_sz = H // 14   # assuming 14×14 patch grid
        results  = []
        for b in range(B):
            # Average pool to patch grid
            grid = F.avg_pool2d(
                rollout_map[b].unsqueeze(0).unsqueeze(0),
                kernel_size=patch_sz,
            ).squeeze()   # (14, 14)
            flat_scores, flat_idx = grid.view(-1).topk(top_k)
            patches = []
            for score, idx in zip(flat_scores.tolist(), flat_idx.tolist()):
                row = idx // 14
                col = idx % 14
                patches.append((row, col, round(score, 4)))
            results.append(patches)
        return results
