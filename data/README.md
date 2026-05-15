# data/README.md
## ADNI Data Preparation

### Step 1 — Apply for ADNI Access
1. Register at https://adni.loni.usc.edu
2. Complete the Data Use Agreement
3. Download T1-weighted MRI scans (ADNI-1, ADNI-GO, ADNI-2, ADNI-3)

### Step 2 — Organise Raw Data
```
raw_adni/
  CN/   {SUBJECT_ID}/  *.nii.gz
  EMCI/ {SUBJECT_ID}/  *.nii.gz
  LMCI/ {SUBJECT_ID}/  *.nii.gz
  AD/   {SUBJECT_ID}/  *.nii.gz
```

### Step 3 — Run Preprocessing
```bash
python scripts/preprocess_adni.py \
    --input_dir  raw_adni \
    --output_dir data/processed \
    --mni_template $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz \
    --n_workers 8 --seed 42
```

### Expected Output
```
data/processed/
  train/ CN/ EMCI/ LMCI/ AD/   # .pt slice tensors
  val/   CN/ EMCI/ LMCI/ AD/
  test/  CN/ EMCI/ LMCI/ AD/
```

**Note:** Preprocessed data cannot be redistributed per ADNI Data Use Agreement.
