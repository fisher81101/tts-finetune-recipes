#!/usr/bin/env python3
"""
Synthesize speech with a VoxCPM LoRA checkpoint.

Usage:
    python infer.py \
        --lora_ckpt output/lora_run1/latest \
        --text "Hello, this is my fine-tuned voice." \
        --out result.wav

Voice cloning (pass a reference clip + its transcript):
    python infer.py \
        --lora_ckpt output/lora_run1/latest \
        --text "This is voice cloning result." \
        --prompt_audio dataset/audio/audio1.wav \
        --prompt_text "Reference audio transcript" \
        --out clone.wav

base_model and LoRA hyperparameters are read straight out of the
checkpoint's lora_config.json — no need to repeat them on the CLI.
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig


def parse_args():
    p = argparse.ArgumentParser("VoxCPM LoRA inference")
    p.add_argument("--lora_ckpt", required=True,
                   help="Checkpoint dir (contains lora_weights.safetensors + lora_config.json)")
    p.add_argument("--base_model", default="",
                   help="Override base model id/path (default: read from lora_config.json)")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--prompt_audio", default="", help="Optional reference WAV for voice cloning")
    p.add_argument("--prompt_text", default="", help="Transcript of the reference WAV")
    p.add_argument("--out", default="output.wav", help="Output WAV path")
    p.add_argument("--cfg_value", type=float, default=2.0)
    p.add_argument("--inference_timesteps", type=int, default=10)
    p.add_argument("--max_len", type=int, default=600)
    p.add_argument("--normalize", action="store_true", help="Enable text normalization")
    p.add_argument("--no_lora", action="store_true", help="Generate with the adapter disabled (A/B baseline)")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt_dir = Path(args.lora_ckpt)
    cfg_path = ckpt_dir / "lora_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"lora_config.json not found in {ckpt_dir}")

    lora_info = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_model = args.base_model or lora_info.get("base_model")
    if not base_model:
        raise ValueError("No base_model in lora_config.json — pass --base_model explicitly")

    lora_cfg_dict = lora_info.get("lora_config", {})
    lora_cfg = LoRAConfig(**lora_cfg_dict) if lora_cfg_dict else None

    print(f"Base model: {base_model}", file=sys.stderr)
    print(f"LoRA ckpt:  {ckpt_dir}", file=sys.stderr)
    if lora_cfg:
        print(f"LoRA cfg:   r={lora_cfg.r} alpha={lora_cfg.alpha}", file=sys.stderr)

    model = VoxCPM.from_pretrained(
        hf_model_id=base_model,
        load_denoiser=False,
        optimize=True,
        lora_config=lora_cfg,
        lora_weights_path=str(ckpt_dir),
    )

    if args.no_lora:
        model.set_lora_enabled(False)

    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=args.prompt_audio or None,
        prompt_text=args.prompt_text or None,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio_np, model.tts_model.sample_rate)
    print(f"Saved {out_path} ({len(audio_np) / model.tts_model.sample_rate:.2f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
