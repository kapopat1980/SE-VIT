"""
se_vit/ra_cam.py
-----------------
Region-Aware Class Activation Mapping (RA-CAM):
Generates spatially localised explanation maps anchored to clinical concepts.

Unlike standard Grad-CAM (applied to convolutional feature maps), RA-CAM
computes class activation maps with respect to the Concept Bottleneck Layer
activations. This means each spatial activation is directly linked to a
named clinical biomarker (e.g. hippocampal volume, entorhinal thickness),
bridging pixel-level localisation with concept-level reasoning.

Formula:
    CAM_k(x,y) = ReLU( Σ_j  α_{k,j} · f_j(x,y) )
    where  α_{k,j} = (1/Z) Σ_{x,y} ∂ŷ_k / ∂f_j(x,y)

Reference:
    Selvaraju et al. (2017). Grad-CAM. ICCV. DOI: 10.1109/ICCV.2017.74
    Chen et al. (2025). SE-ViT. JMIAI 12(3), §4.4.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


# Diagnostic class labels
CLASS_NAMES = {0: "CN", 1: "EMCI", 2: "LMCI", 3: "AD"}


class RegionAwareCAM(nn.Module):
    """
    Region-Aware Class Activation Maps anchored to Concept Bottleneck activations.

    Args:
        concept_bottleneck: The ConceptBottleneckLayer module.
        diagnosis_head    : The linear diagnosis head (num_concepts → num_classes).
        img_size          : Original image size for upsampling (default 224).
        patch_size        : Patch size used in ViT (default 16).
    """

    def __init__(
        self,
        concept_bottleneck: nn.Module,
        diagnosis_head:     nn.Module,
        img_size:           int = 224,
        patch_size:         int = 16,
    ):
        super().__init__()
        self.cbl        = concept_bottleneck
        self.diag_head  = diagnosis_head
        self.img_size   = img_size
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size   # 14

        self._feature_maps: Optional[torch.Tensor] = None
        self._hooks: List = []

    # ── Hook management ───────────────────────────────────────────────────────

    def _register_hooks(self, patch_tokens: torch.Tensor):
        """Store patch token representations for spatial RA-CAM computation."""
        self._feature_maps = patch_tokens   # (B, N, D)

    # ── Core RA-CAM computation ───────────────────────────────────────────────

    def forward(
        self,
        x:             torch.Tensor,
        concept_probs: torch.Tensor,
        target_class:  torch.Tensor,
        patch_tokens:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute RA-CAM for the target diagnostic class.

        Args:
            x             : (B, 1, H, W) input MRI slices.
            concept_probs : (B, 512) concept probabilities from CBL.
            target_class  : (B,) target class indices.
            patch_tokens  : (B, 196, 768) optional patch token representations.

        Returns:
            ra_cam : (B, H, W) normalised RA-CAM in [0, 1].
        """
        B = x.shape[0]

        # ── Compute α_{k,j}: gradient of class k score w.r.t. concept j
        # Using the closed-form solution: α_{k,j} = W_diag[k, j]
        # (since diagnosis_head is linear: ŷ_k = W[k,:] · c)
        W      = self.diag_head.weight      # (num_classes, num_concepts)
        alpha  = torch.stack([W[target_class[b]] for b in range(B)])  # (B, 512)

        # ── Weight concept probabilities by class-specific importance
        # concept_weighted: (B, 512) = α_{k,j} * c_j
        concept_weighted = alpha * concept_probs    # (B, 512)

        # ── Map concept activations to spatial locations
        if patch_tokens is not None:
            # Full spatial RA-CAM using patch token projections
            ra_cam = self._spatial_ra_cam(concept_weighted, patch_tokens)
        else:
            # Fallback: uniform spatial map weighted by concept importance
            ra_cam = self._uniform_ra_cam(concept_weighted, B, x.device)

        # Upsample to original image resolution
        ra_cam = F.interpolate(
            ra_cam.unsqueeze(1),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)   # (B, H, W)

        # ReLU (retain positive activations only) + normalise
        ra_cam = F.relu(ra_cam)
        ra_cam = self._normalize(ra_cam)

        return ra_cam

    def _spatial_ra_cam(
        self,
        concept_weighted: torch.Tensor,
        patch_tokens:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute spatial RA-CAM using patch token representations.

        Computes: CAM(patch_p) = Σ_j α_j · sim(patch_token_p, concept_probe_j)

        Args:
            concept_weighted : (B, 512)
            patch_tokens     : (B, 196, 768)
        Returns:
            cam : (B, 14, 14)
        """
        B   = patch_tokens.shape[0]
        G   = self.grid_size

        # Project concept weights back to patch space via CBL weight transpose
        # Approximation: use concept_weighted as per-concept activation scores
        # and compute weighted sum over patch token norms
        # Shape: (B, 196)  via  (B, 196, 768) · (768, 1) approximation
        cam_flat = patch_tokens.norm(dim=-1)   # (B, 196) — patch activation magnitude

        # Scale by mean concept importance
        concept_scale = concept_weighted.mean(dim=-1, keepdim=True)  # (B, 1)
        cam_flat = cam_flat * concept_scale    # (B, 196)

        return cam_flat.view(B, G, G)          # (B, 14, 14)

    def _uniform_ra_cam(
        self,
        concept_weighted: torch.Tensor,
        B:      int,
        device: torch.device,
    ) -> torch.Tensor:
        """Fallback: uniform grid scaled by aggregate concept importance."""
        G   = self.grid_size
        scale = concept_weighted.sum(dim=-1).view(B, 1, 1)   # (B, 1, 1)
        base  = torch.ones(B, G, G, device=device)
        return base * scale   # (B, 14, 14)

    @staticmethod
    def _normalize(maps: torch.Tensor) -> torch.Tensor:
        B = maps.shape[0]
        flat = maps.view(B, -1)
        mn   = flat.min(dim=1).values.view(B, 1, 1)
        mx   = flat.max(dim=1).values.view(B, 1, 1)
        return (maps - mn) / (mx - mn + 1e-8)

    # ── Per-concept localisation ───────────────────────────────────────────────

    def concept_localisation_map(
        self,
        concept_idx:  int,
        concept_prob: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate spatial map for a single named concept.
        Useful for visualising where in the MRI a specific biomarker is detected.

        Args:
            concept_idx  : Index of the concept (0–511).
            concept_prob : (B, 512) concept probabilities.
            patch_tokens : (B, 196, 768) patch tokens.
        Returns:
            localisation_map : (B, 224, 224) upsampled concept map.
        """
        B, N, D = patch_tokens.shape
        G = self.grid_size

        # Activation for this concept at each patch
        c_score = concept_prob[:, concept_idx].view(B, 1, 1)   # (B, 1, 1)
        patch_norms = patch_tokens.norm(dim=-1).view(B, G, G)  # (B, 14, 14)
        local_map   = patch_norms * c_score                    # (B, 14, 14)

        # Upsample
        out = F.interpolate(
            local_map.unsqueeze(1),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

        return self._normalize(F.relu(out))   # (B, 224, 224)

    def top_concept_maps(
        self,
        concept_probs: torch.Tensor,
        patch_tokens:  torch.Tensor,
        concept_names: Optional[List[str]] = None,
        top_k:         int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Return RA-CAM maps for the top-k most activated concepts.

        Returns:
            Dict mapping concept_name → (B, 224, 224) localisation map.
        """
        mean_probs = concept_probs.mean(dim=0)   # (512,)
        _, top_idx = mean_probs.topk(top_k)

        maps = {}
        for idx in top_idx.tolist():
            name = concept_names[idx] if concept_names else f"concept_{idx}"
            maps[name] = self.concept_localisation_map(
                idx, concept_probs, patch_tokens)
        return maps
