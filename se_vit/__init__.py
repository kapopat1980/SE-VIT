"""
SE-ViT: Self-Explaining Vision Transformer for Alzheimer's Disease Diagnosis
============================================================================

Official implementation of:
    Chen et al. (2025). Developing Trustworthy AI: A Self-Explaining Vision
    Transformer (ViT) Architecture for Analyzing ADNI MRI Biomarkers.
    JMIAI 12(3), 412–438. DOI: 10.1016/j.jmiai.2025.04.0042

Quick start:
    from se_vit import SEViT
    model = SEViT.from_pretrained('sevit_adni_best.pth')
    output = model.explain(mri_slice)
    print(output.probabilities)          # (B, 4) — CN, EMCI, LMCI, AD
    print(output.concept_attributions)   # top clinical concepts
"""

from .model             import SEViT, SEViTOutput
from .concept_bottleneck import ConceptBottleneckLayer, CONCEPT_CATEGORIES
from .gradient_rollout  import GradientRollout
from .ra_cam            import RegionAwareCAM
from .losses            import SEViTLoss, TemperatureScaling, expected_calibration_error

__version__  = "1.0.0"
__authors__  = ["Wei Chen", "Priya Nair", "James O'Sullivan", "Amara Diallo"]
__license__  = "MIT"
__paper_doi__ = "10.1016/j.jmiai.2025.04.0042"

__all__ = [
    "SEViT",
    "SEViTOutput",
    "ConceptBottleneckLayer",
    "CONCEPT_CATEGORIES",
    "GradientRollout",
    "RegionAwareCAM",
    "SEViTLoss",
    "TemperatureScaling",
    "expected_calibration_error",
]
