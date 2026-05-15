"""
scripts/preprocess_adni.py
---------------------------
ADNI T1-weighted MRI preprocessing pipeline.

Applies the 8-step preprocessing pipeline described in:
    Chen et al. (2025). SE-ViT. JMIAI 12(3), §3.3.

Steps:
    1. AC-PC alignment        (FreeSurfer v7.3.2)
    2. Brain extraction       (BET v2.1)
    3. Bias field correction  (N4ITK / ANTs v2.4.3)
    4. MNI152 registration    (ANTs SyN)
    5. Tissue segmentation    (FSL FAST v6.0.4)
    6. GM density modulation  (Jacobian modulation)
    7. Intensity normalisation (z-score within brain mask)
    8. Slice extraction       (48 axial slices, 224×224)

Requirements:
    FreeSurfer 7.3.2, FSL 6.0.4, ANTs 2.4.3 must be installed and
    available on PATH. Python deps: nibabel, nilearn, numpy, torch.

Usage:
    python scripts/preprocess_adni.py \\
        --input_dir  /path/to/raw_adni \\
        --output_dir data/processed \\
        --n_workers  8 \\
        --seed       42
"""

import os
import sys
import argparse
import random
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import nibabel as nib
    import torch
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("[Warning] nibabel/torch not found — install with: pip install nibabel torch")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_MAP   = {"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}
IMG_SIZE    = 224
N_SLICES    = 48     # axial slices centred on hippocampal body
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Individual preprocessing steps
# ─────────────────────────────────────────────────────────────────────────────

def step1_acpc_align(nii_path: Path, out_dir: Path) -> Path:
    """AC-PC alignment using FreeSurfer recon-all."""
    out_path = out_dir / "acpc_aligned.nii.gz"
    if out_path.exists():
        return out_path
    cmd = [
        "mri_convert",
        "--conform",
        str(nii_path),
        str(out_path),
    ]
    _run(cmd, "AC-PC alignment")
    return out_path


def step2_brain_extraction(nii_path: Path, out_dir: Path) -> Path:
    """Brain extraction using FSL BET v2.1."""
    out_path  = out_dir / "brain_extracted"
    mask_path = out_dir / "brain_mask.nii.gz"
    if Path(str(out_path) + ".nii.gz").exists():
        return Path(str(out_path) + ".nii.gz")
    cmd = [
        "bet", str(nii_path), str(out_path),
        "-f", "0.5",        # fractional intensity threshold
        "-m",               # output brain mask
        "-R",               # robust brain centre estimation
    ]
    _run(cmd, "Brain extraction (BET v2.1)")
    return Path(str(out_path) + ".nii.gz")


def step3_bias_correction(nii_path: Path, out_dir: Path) -> Path:
    """N4ITK bias field correction using ANTs v2.4.3."""
    out_path = out_dir / "n4_corrected.nii.gz"
    if out_path.exists():
        return out_path
    cmd = [
        "N4BiasFieldCorrection",
        "-d", "3",
        "-i", str(nii_path),
        "-o", str(out_path),
        "-c", "[50x50x50x50,0.0001]",
        "-s", "4",
        "-b", "[180]",
        "-t", "[0.3,0.01,200]",
    ]
    _run(cmd, "N4ITK bias field correction (ANTs v2.4.3)")
    return out_path


def step4_mni_registration(nii_path: Path, out_dir: Path,
                            mni_template: str) -> Path:
    """Affine + SyN nonlinear registration to MNI152 using ANTs."""
    prefix   = str(out_dir / "mni_reg_")
    out_path = out_dir / "mni_registered.nii.gz"
    if out_path.exists():
        return out_path
    cmd = [
        "antsRegistrationSyN.sh",
        "-d", "3",
        "-f", mni_template,
        "-m", str(nii_path),
        "-o", prefix,
        "-t", "a",    # affine only; use 's' for SyN (slower)
        "-n", "4",    # number of threads
    ]
    _run(cmd, "MNI152 registration (ANTs SyN)")
    # Rename output
    warped = out_dir / "mni_reg_Warped.nii.gz"
    if warped.exists():
        warped.rename(out_path)
    return out_path


def step5_tissue_segmentation(nii_path: Path, out_dir: Path) -> Path:
    """GM/WM/CSF segmentation using FSL FAST v6.0.4."""
    prefix   = str(out_dir / "fast_seg")
    gm_path  = out_dir / "fast_seg_pve_1.nii.gz"   # PVE class 1 = GM
    if gm_path.exists():
        return gm_path
    cmd = [
        "fast",
        "-t", "1",          # T1-weighted
        "-n", "3",          # 3 tissue classes
        "-H", "0.1",
        "-I", "4",
        "-l", "20.0",
        "-o", prefix,
        str(nii_path),
    ]
    _run(cmd, "Tissue segmentation (FSL FAST v6.0.4)")
    return gm_path


def step6_gm_modulation(gm_path: Path, jacobian_path: Path,
                        out_dir: Path) -> Path:
    """Apply Jacobian modulation to GM partial volume estimate."""
    if not HAS_DEPS:
        return gm_path
    out_path = out_dir / "gm_modulated.nii.gz"
    if out_path.exists():
        return out_path

    gm_img  = nib.load(str(gm_path))
    jac_img = nib.load(str(jacobian_path))
    gm_data  = gm_img.get_fdata()
    jac_data = jac_img.get_fdata()

    # Modulated GM = GM × |J|
    mod_gm = gm_data * np.abs(jac_data)
    nib.save(nib.Nifti1Image(mod_gm, gm_img.affine), str(out_path))
    return out_path


def step7_intensity_normalise(nii_path: Path, out_dir: Path,
                              mask_path: Path = None) -> Path:
    """Z-score normalisation within brain mask."""
    if not HAS_DEPS:
        return nii_path
    out_path = out_dir / "normalised.nii.gz"
    if out_path.exists():
        return out_path

    img  = nib.load(str(nii_path))
    data = img.get_fdata()

    if mask_path and mask_path.exists():
        mask = nib.load(str(mask_path)).get_fdata() > 0.5
    else:
        mask = data > data.mean() * 0.1

    mu  = data[mask].mean()
    std = data[mask].std() + 1e-8
    normalised = (data - mu) / std
    nib.save(nib.Nifti1Image(normalised, img.affine), str(out_path))
    return out_path


def step8_extract_slices(nii_path: Path, out_dir: Path,
                         subject_id: str, label: int) -> int:
    """
    Extract 48 axial slices centred on the hippocampal body,
    resize to 224×224, and save as .pt tensors.
    """
    if not HAS_DEPS:
        return 0

    img  = nib.load(str(nii_path))
    data = img.get_fdata()   # (91, 109, 91) after 2mm resampling
    Z    = data.shape[2]

    # Centre on axial slice 45 (hippocampal body at ~MNI z=-18mm)
    centre   = Z // 2
    half     = N_SLICES // 2
    z_start  = max(0, centre - half)
    z_end    = min(Z, z_start + N_SLICES)
    n_saved  = 0

    for i, z in enumerate(range(z_start, z_end)):
        slice_data = data[:, :, z].astype(np.float32)

        # Resize to 224×224
        from PIL import Image
        pil_img    = Image.fromarray(slice_data).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        tensor     = torch.tensor(np.array(pil_img)).unsqueeze(0)  # (1, 224, 224)

        # Save
        out_path   = out_dir / f"{subject_id}_z{z:03d}.pt"
        torch.save(tensor, str(out_path))
        n_saved   += 1

    return n_saved


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline per subject
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_subject(args_tuple):
    nii_path, out_dir, subject_id, label, mni_template = args_tuple

    work_dir = out_dir / "work" / subject_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        p = nii_path
        p = step1_acpc_align(p, work_dir)
        p = step2_brain_extraction(p, work_dir)
        p = step3_bias_correction(p, work_dir)
        if mni_template and Path(mni_template).exists():
            p = step4_mni_registration(p, work_dir, mni_template)
        p = step5_tissue_segmentation(p, work_dir)
        p = step7_intensity_normalise(p, work_dir)
        n = step8_extract_slices(p, out_dir, subject_id, label)
        return subject_id, True, n
    except Exception as e:
        print(f"  [Error] Subject {subject_id}: {e}")
        return subject_id, False, 0


def _run(cmd, step_name):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [Warning] {step_name}: {result.stderr[:200]}")
    except FileNotFoundError:
        print(f"  [Skip] {step_name}: tool not found on PATH")
    except Exception as e:
        print(f"  [Error] {step_name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset splitting
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(subject_list, seed=42):
    """
    Subject-level stratified split: 70% train / 15% val / 15% test.
    Preserves class distribution. Ensures no subject appears in >1 split.
    """
    random.seed(seed)
    np.random.seed(seed)

    by_class = {}
    for subj_id, label, pt_files in subject_list:
        by_class.setdefault(label, []).append((subj_id, label, pt_files))

    train, val, test = [], [], []
    for label, subjects in by_class.items():
        random.shuffle(subjects)
        n       = len(subjects)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        train  += subjects[:n_train]
        val    += subjects[n_train:n_train + n_val]
        test   += subjects[n_train + n_val:]

    print(f"\n[Split] Train={len(train)} | Val={len(val)} | Test={len(test)} subjects")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Discover subjects
    # Expected structure: input_dir/{DIAGNOSIS}/{SUBJECT_ID}/*.nii.gz
    subjects = []
    for class_name, label in CLASS_MAP.items():
        class_dir = input_dir / class_name
        if not class_dir.exists():
            print(f"[Warning] {class_dir} not found — skipping")
            continue
        for subj_dir in sorted(class_dir.iterdir()):
            nii_files = list(subj_dir.glob("*.nii.gz")) + \
                        list(subj_dir.glob("*.nii"))
            if nii_files:
                subjects.append((nii_files[0], subj_dir.name, label))

    print(f"[Preprocess] Found {len(subjects)} subjects in {input_dir}")

    # Subject-level split BEFORE processing (prevent leakage)
    subject_meta = [(sid, lbl, None) for _, sid, lbl in subjects]
    train_meta, val_meta, test_meta = stratified_split(subject_meta, seed=args.seed)
    split_map = {}
    for sid, lbl, _ in train_meta: split_map[sid] = "train"
    for sid, lbl, _ in val_meta:   split_map[sid] = "val"
    for sid, lbl, _ in test_meta:  split_map[sid] = "test"

    # Create output split/class directories
    for split in ["train", "val", "test"]:
        for class_name in CLASS_MAP:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    # Build processing tasks
    tasks = []
    for nii_path, subject_id, label in subjects:
        split      = split_map.get(subject_id, "train")
        class_name = {v: k for k, v in CLASS_MAP.items()}[label]
        out_subdir = output_dir / split / class_name
        tasks.append((nii_path, out_subdir, subject_id, label, args.mni_template))

    # Process subjects (parallel)
    ok = fail = total_slices = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(preprocess_subject, t): t for t in tasks}
        for fut in as_completed(futures):
            subj_id, success, n_slices = fut.result()
            if success:
                ok           += 1
                total_slices += n_slices
                print(f"  ✅ {subj_id}: {n_slices} slices saved")
            else:
                fail += 1

    print(f"\n{'='*55}")
    print(f"Preprocessing complete!")
    print(f"  Subjects processed : {ok}/{ok+fail}")
    print(f"  Total slices saved : {total_slices}")
    print(f"  Output directory   : {output_dir}")
    print(f"{'='*55}")
    print("\nNext step: python scripts/train.py --config configs/sevit_base.yaml "
          f"--data_dir {output_dir} --output_dir outputs/sevit_run1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ADNI MRI data for SE-ViT training")
    parser.add_argument("--input_dir",    required=True,
                        help="Root directory of raw ADNI NIfTI files")
    parser.add_argument("--output_dir",   required=True,
                        help="Output directory for processed .pt slice tensors")
    parser.add_argument("--mni_template", default="",
                        help="Path to MNI152 T1 2mm template NIfTI "
                             "(e.g. $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz)")
    parser.add_argument("--n_workers",    type=int, default=4,
                        help="Number of parallel workers (default 4)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()
    main(args)
