#!/usr/bin/env python3
"""
Thin launcher for VoxCPM LoRA fine-tuning.

Usage:
    python train.py --args.load=config.yaml

All actual training logic lives in train_voxcpm_finetune.py (vendored from
the upstream VoxCPM repo, see Credits in README.md). This wrapper just keeps
the entry point inside the recipe folder so you don't need to clone VoxCPM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_voxcpm_finetune import train  # noqa: E402

if __name__ == "__main__":
    train()
