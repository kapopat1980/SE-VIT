"""
se_vit/model.py
---------------
Core SE-ViT architecture: Self-Explaining Vision Transformer for ADNI MRI classification.

Reference:
    Chen et al. (2025). Developing Trustworthy AI: A Self-Explaining Vision Transformer
    Architecture for Analyzing ADNI MRI Biomarkers. JMIAI 12(3), 412–438.
    DOI: 10.1016/j.jmiai.2025.04.0042
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .concept_bottleneck import ConceptBottleneckLayer
from .gradient_rollout import GradientRollout
from .ra_cam import RegionAwareCAM


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SEViTOutput:
    """All outputs produced by SE-ViT in a single forward pass."""
    logits: torch.Tensor                        # (B, 4) diagnostic logits
    probabilities: torch.Tensor                 # (B, 4) softmax probabilities
    concept_probs: torch.Tensor                 # (B, 512) concept probabilities
    concept_attributions: Optional[Dict] = None # concept → contribution score
    gradient_rollout_map: Optional[torch.Tensor] = None  # (B, H, W)
    ra_cam: Optional[torch.Tensor] = None                # (B, H, W)
    uncertainty: Optional[torch.Tensor] = None           # (B,) entropy


# ─────────────────────────────────────────────────────────────────────────────
# Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 1, embed_dim: int = 768):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2   # 196
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  →  (B, n_patches, embed_dim)
        x = self.proj(x)                      # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Store attention weights for gradient rollout
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)     # (3, B, H, N, d_k)
        q, k, v = qkv.unbind(0)               # each (B, H, N, d_k)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Retain for gradient rollout
        self.attention_weights = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj_drop(self.proj(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MLP Block
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):

    def __init__(self, embed_dim: int = 768, mlp_ratio: float = 4.0,
                 drop: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1  = nn.Linear(embed_dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop: float = 0.0,
                 attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads,
                                            attn_drop=attn_drop,
                                            proj_drop=drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SE-ViT Main Model
# ─────────────────────────────────────────────────────────────────────────────

class SEViT(nn.Module):
    """
    Self-Explaining Vision Transformer for 4-class ADNI MRI classification.

    Architecture:
        PatchEmbedding → 12× TransformerBlock → ConceptBottleneckLayer
        → DiagnosisHead  +  ExplanationHead (GradRollout + RA-CAM)

    Args:
        img_size      (int):   Input image size (default 224).
        patch_size    (int):   Patch size (default 16 → 196 patches).
        in_channels   (int):   Input channels — 1 for grayscale MRI.
        embed_dim     (int):   Transformer embedding dimension (default 768).
        depth         (int):   Number of Transformer blocks (default 12).
        num_heads     (int):   Attention heads per block (default 12).
        mlp_ratio     (float): MLP hidden dimension ratio (default 4.0).
        num_classes   (int):   Diagnostic classes: CN/EMCI/LMCI/AD (default 4).
        num_concepts  (int):   Clinical concept dimensions in CBL (default 512).
        drop_rate     (float): Dropout rate (default 0.1).
        temperature   (float): Calibration temperature (default 1.42, fitted on val set).
    """

    def __init__(
        self,
        img_size:     int   = 224,
        patch_size:   int   = 16,
        in_channels:  int   = 1,
        embed_dim:    int   = 768,
        depth:        int   = 12,
        num_heads:    int   = 12,
        mlp_ratio:    float = 4.0,
        num_classes:  int   = 4,
        num_concepts: int   = 512,
        drop_rate:    float = 0.1,
        temperature:  float = 1.42,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.num_classes  = num_classes
        self.num_concepts = num_concepts
        self.temperature  = temperature

        # ── Patch embedding + positional encoding
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, embed_dim))  # +1 for [CLS]
        self.pos_drop  = nn.Dropout(drop_rate)

        # ── Transformer encoder (12 layers)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio,
                             drop=drop_rate, attn_drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ── Concept Bottleneck Layer
        self.cbl = ConceptBottleneckLayer(embed_dim, num_concepts)

        # ── Diagnosis head  (concept space → 4 classes)
        self.diagnosis_head = nn.Linear(num_concepts, num_classes, bias=True)

        # ── Explainability modules
        self.gradient_rollout = GradientRollout(self.blocks)
        self.ra_cam           = RegionAwareCAM(self.cbl, self.diagnosis_head)

        # ── Weight initialisation
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run patch embedding + positional encoding + transformer."""
        B = x.shape[0]
        x = self.patch_embed(x)                         # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                # (B, N+1, D)
        x   = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]   # [CLS] token: (B, D)

    def forward(
        self,
        x: torch.Tensor,
        return_explanations: bool = False,
        concept_labels: Optional[torch.Tensor] = None,
    ) -> SEViTOutput:
        """
        Args:
            x                   : (B, 1, 224, 224) grayscale MRI slices.
            return_explanations : If True, compute gradient rollout + RA-CAM.
            concept_labels      : (B, 512) binary concept annotations for CBL loss.

        Returns:
            SEViTOutput with logits, probabilities, concept_probs, and
            optionally concept_attributions, gradient_rollout_map, ra_cam,
            uncertainty.
        """
        cls_repr    = self.forward_features(x)               # (B, 768)
        concept_prob= self.cbl(cls_repr)                     # (B, 512)
        logits      = self.diagnosis_head(concept_prob)      # (B, 4)
        cal_logits  = logits / self.temperature
        probs       = F.softmax(cal_logits, dim=-1)          # (B, 4)

        # Entropy-based uncertainty
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # (B,)

        output = SEViTOutput(
            logits=logits,
            probabilities=probs,
            concept_probs=concept_prob,
            uncertainty=entropy,
        )

        if return_explanations:
            pred_class = probs.argmax(dim=-1)   # (B,)

            # Concept attribution: W[k,j] * c_j for predicted class k
            W = self.diagnosis_head.weight      # (4, 512)
            attributions = {}
            for b in range(x.shape[0]):
                k = pred_class[b].item()
                scores = (W[k] * concept_prob[b]).detach().cpu()
                attributions[b] = {
                    f"concept_{j}": scores[j].item()
                    for j in range(self.num_concepts)
                }
            output.concept_attributions = attributions

            # Gradient rollout attention map
            output.gradient_rollout_map = self.gradient_rollout(
                x, pred_class)   # (B, 224, 224)

            # RA-CAM
            output.ra_cam = self.ra_cam(
                x, concept_prob, pred_class)   # (B, 224, 224)

        return output

    def explain(self, x: torch.Tensor) -> SEViTOutput:
        """Convenience wrapper: forward pass with full explanations enabled."""
        return self.forward(x, return_explanations=True)

    # ── Class methods ─────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "SEViT":
        """Load pretrained SE-ViT weights."""
        model = cls(**kwargs)
        state = torch.load(checkpoint_path, map_location="cpu")
        # Support both raw state_dict and checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[SEViT] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"[SEViT] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        model.eval()
        return model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
