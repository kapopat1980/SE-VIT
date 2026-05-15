# docs/model_card.md  ── SE-ViT Model Card

## Model Details
- **Name:** SE-ViT (Self-Explaining Vision Transformer)
- **Version:** 1.0.0
- **Type:** Vision Transformer + Concept Bottleneck Layer
- **Task:** 4-class MRI classification (CN / EMCI / LMCI / AD)
- **Paper:** Chen et al. (2025), JMIAI 12(3), DOI: 10.1016/j.jmiai.2025.04.0042
- **License:** MIT

## Intended Use
- **Primary use:** AI-assisted decision support for Alzheimer's disease staging
- **Intended users:** Neuroradiologists, neurologists, neuroimaging researchers
- **Out-of-scope:** Standalone diagnostic tool; non-MRI modalities

## Training Data
- **Dataset:** ADNI-1, ADNI-GO, ADNI-2, ADNI-3
- **n:** 1,118 subjects (419 CN, 337 EMCI, 154 LMCI, 208 AD)
- **Modality:** T1-weighted structural MRI
- **Preprocessing:** FreeSurfer 7.3.2 + BET + ANTs + FSL

## Performance (Test Set, n=167 subjects)

| Metric | Value |
|---|---|
| Overall Accuracy | 91.2 ± 0.9% |
| AUC-ROC (macro) | 0.954 |
| Sensitivity | 0.897 |
| Specificity | 0.923 |
| ECE (Calibration) | 0.034 |
| Expert Usefulness (1–5) | 4.5 / 5.0 |

## Fairness

| Subgroup | AUC-ROC | EOD |
|---|---|---|
| Female | 0.951 | < 0.05 |
| Male | 0.957 | < 0.05 |
| Age ≤70 | 0.961 | < 0.05 |
| Age 71–76 | 0.952 | < 0.05 |
| Age ≥77 | 0.947 | < 0.05 |

## Limitations
1. Trained on ADNI (predominantly White non-Hispanic, academic centre patients)
2. 2D slice-based input — volumetric 3D ViT extension recommended
3. Single-site concept annotation validation (n=200 subjects)
4. No external validation on non-ADNI cohorts (OASIS-3, UK Biobank)
5. Not validated as a standalone diagnostic device

## Ethical Considerations
- ADNI IRB approval: UCSF Protocol 10-02245
- Expert evaluation IRB: MGH-2024-0371
- Model outputs are intended as decision support, not autonomous diagnosis
- Uncertainty estimates should trigger mandatory specialist review
