# Training Machine Setup & Execution Guide

This guide sets up the Network World Model training pipeline on a separate GPU machine (RTX 4060, 4090, etc.).

## Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (RTX 4060+)
- **Python**: 3.10+
- **CUDA**: 11.8+ with cuDNN
- **Storage**: 50GB for CIC-IDS2017 dataset + models

## Step 1: Environment Setup

```bash
# Clone/copy repository to training machine
git clone <repo-url> world-model-ids
cd world-model-ids

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Step 2: Dataset Preparation

```bash
# Download CIC-IDS2017 from https://www.unb.ca/research/iscxdownload/
# Extract CSVs to a folder (e.g., /data/cic_ids2017/)

# Prepare dataset (one-time)
python -m src.data.prepare \
  --input-dir /data/cic_ids2017 \
  --output data/raw/flows.csv \
  --source cic_ids2017

# Expected output:
#   [prepare] Loaded 2,830,743 raw rows
#   [prepare] ✓ All required columns present
#   [prepare] Cleaning stats: ...
#   [prepare] ✓ Saved processed flows: data/raw/flows.csv
#   [prepare] ✓ Saved provenance: data/raw/flows_provenance.json
```

## Step 3: Training with Progress Monitoring

### Option A: Simple Training Script

```bash
# Single command training with all outputs
python -m src.train --config configs/default.yaml 2>&1 | tee training.log

# Monitor progress in real-time:
tail -f training.log | grep -E "epoch|selected threshold|best checkpoint"
```

### Option B: Advanced Monitoring Script

Save this as `train_monitor.py`:

```python
#!/usr/bin/env python
"""Monitor training progress with live updates"""
import subprocess
import sys
import time
import re
from pathlib import Path

def monitor_training():
    # Start training process
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.train", "--config", "configs/default.yaml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Tracking variables
    best_val_loss = float('inf')
    epoch_pattern = re.compile(r'epoch\s+(\d+).*train_loss=([\d.]+) val_loss=([\d.]+)')
    best_pattern = re.compile(r'loading best checkpoint from epoch (\d+)')
    threshold_pattern = re.compile(r'selected threshold=([\d.]+)')
    
    print("="*70)
    print("TRAINING MONITOR - Network World Model")
    print("="*70)
    print()
    
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        
        # Parse epoch progress
        match = epoch_pattern.search(line)
        if match:
            epoch, train_loss, val_loss = match.groups()
            val_loss_f = float(val_loss)
            if val_loss_f < best_val_loss:
                best_val_loss = val_loss_f
                print(f"  ✓ Best epoch: {epoch}, val_loss: {val_loss_f:.4f}", file=sys.stderr)
        
        # Track best checkpoint
        if "best checkpoint" in line:
            print("\n[✓] Best checkpoint loaded for threshold selection", file=sys.stderr)
        
        # Track threshold selection
        if "selected threshold" in line:
            match = threshold_pattern.search(line)
            if match:
                threshold = match.group(1)
                print(f"\n[✓] Threshold selected: {threshold}", file=sys.stderr)
    
    proc.wait()
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  python -m evaluation.benchmark")
    print("  python -m src.rollout")
    print("  python -m src.explain")
    return proc.returncode

if __name__ == "__main__":
    sys.exit(monitor_training())
```

Run it:
```bash
python train_monitor.py
```

### Option C: Background Training with TensorBoard (Advanced)

```bash
# Install TensorBoard logging
pip install tensorboard

# Add to src/train.py (line ~290):
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/experiment")
# writer.add_scalar("train/loss", loss, epoch)
# writer.add_scalar("val/loss", val_loss, epoch)

# Train in background
nohup python -m src.train --config configs/default.yaml > training.log 2>&1 &

# Monitor in real-time
tensorboard --logdir=runs/experiment --port=6006
# View at http://localhost:6006
```

## Step 4: Expected Training Output

```
======================================================================
DATA SOURCE: REAL (CIC-IDS2017)
======================================================================
[train] dataset: CIC-IDS2017
[train] flows retained: 2,830,743
[train] hosts: 234
[train] duration: 4.9 hours

[train] built 58,402 sequences across 234 hosts (12.3% positive infiltration rate)
[train] using per-scenario chronological split (ensures attacks appear in all splits)
[train] ✓ Chronological split verified (no temporal leakage)
[train] ✓ Attack distribution: train=4567, val=1123, test=890

[train] training lstm world model for 60 epochs...
  epoch  0  train_loss=0.8234 val_loss=0.7891  (mse=0.4523 bce=0.2341 ce=0.1027)
  epoch  1  train_loss=0.7156 val_loss=0.7245  (best)
  ...
  epoch 23  train_loss=0.4123 val_loss=0.4456  (no improvement for 5 epochs, early stopping)
[train] loading best checkpoint from epoch 18 (val_loss=0.4234)
[train] selected threshold=0.625 (F1=0.789 on validation)
[train] LR baseline: selected threshold=0.550 (F1=0.654 on validation)
[train] saved: weights/world_model.pt, scaler.pkl, baseline_lr.pkl, ...
```

## Step 5: Evaluate & Generate Results

After training completes:

```bash
# Evaluation on test set
python -m evaluation.benchmark
# Output: evaluation/results.md

# Generate demo rollout (K-step forecast)
python -m src.rollout
# Output: JSON demo + attention trace

# Generate SHAP explanations
python -m src.explain
# Output: feature importance for sample

# View results
cat evaluation/results.md
```

## Step 6: Troubleshooting

### OOM (Out of Memory)
Reduce model size in `configs/default.yaml`:
```yaml
model:
  hidden_dim: 32  # reduce from 64
  num_layers: 1   # reduce from 2
```

### CUDA not found
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, reinstall PyTorch for your CUDA version
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Dataset preparation fails
```bash
# Check if CIC-IDS2017 CSVs have required columns
python -c "
import pandas as pd
df = pd.read_csv('/data/cic_ids2017/Monday.csv')
print('Columns:', df.columns.tolist())
print('Has Src IP:', 'Src IP' in df.columns or 'Source IP' in df.columns)
"
```

### Training hangs
```bash
# Check GPU status
nvidia-smi

# Kill stuck process
pkill -f "src.train"

# Resume (loads from checkpoint if available)
python -m src.train --config configs/default.yaml
```

## Output Artifacts

After training:
```
weights/
  ├── world_model.pt              # Model weights (torch.save)
  ├── scaler.pkl                  # StandardScaler for features
  ├── baseline_lr.pkl             # Logistic regression baseline
  ├── threshold.json              # Frozen threshold
  ├── feature_columns.json        # Feature names
  ├── used_config.yaml            # Configuration used
  ├── training_history.json       # Loss per epoch
  └── validation_metrics.json     # Best validation metrics

data/raw/
  ├── flows.csv                   # Processed CIC-IDS2017
  └── flows_provenance.json       # Dataset metadata

data/processed/
  └── test_split.npz              # Test set for benchmark

evaluation/
  └── results.md                  # Benchmark results table
```

## Performance Metrics

Expected results on CIC-IDS2017 (RTX 4060):
- **Training time**: 20-40 minutes (60 epochs, may early-stop at 15-25)
- **GPU memory**: 4-6 GB
- **Model size**: ~2 MB (world_model.pt)
- **Test F1**: 0.70-0.85 (depending on dataset distribution)
- **Test AUC-ROC**: 0.75-0.90

## Batch Processing Multiple Runs

To compare different configurations:

```bash
#!/bin/bash
# save as train_batch.sh

for hidden_dim in 32 64 128; do
  echo "Training with hidden_dim=$hidden_dim"
  
  # Modify config
  sed -i "s/hidden_dim: .*/hidden_dim: $hidden_dim/" configs/default.yaml
  
  # Train
  python -m src.train --config configs/default.yaml
  
  # Save results
  mkdir -p results/hidden_dim_$hidden_dim
  cp -r weights/* results/hidden_dim_$hidden_dim/
  cp evaluation/results.md results/hidden_dim_$hidden_dim/
  
  echo "Complete: hidden_dim=$hidden_dim"
done
```

Run:
```bash
bash train_batch.sh
```

---

**For questions or issues**, check `training.log` or run:
```bash
python -m src.data.prepare --help
python -m src.train --help
python -m evaluation.benchmark --help
```
