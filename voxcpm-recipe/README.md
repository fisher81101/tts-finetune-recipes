# VoxCPM 1.5 LoRA Fine-Tuning Recipe

Fine-tune **VoxCPM 1.5** — OpenBMB's tokenizer-free, diffusion-based TTS model
(built on MiniCPM-4) — on your own speaker/domain data using LoRA. The result
is a small (~16 MB) adapter that nudges the base model toward your voice and
vocabulary, without touching the full ~1.5B-parameter weights.

> **Example checkpoint trained with this recipe:**
> [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-voxcpm--lora--finetune-ff6b00?style=flat-square)](https://huggingface.co/jeevav62/voxcpm-lora-finetune)
> — VoxCPM 1.5 LoRA tuned on 100 tech-vocabulary clips (rank 16, step 2000).
> Early checkpoints mispronounced symbols like `#` and `.`; by step 2000 the
> model had recovered and rendered them naturally.
>
> 🔗 https://huggingface.co/jeevav62/voxcpm-lora-finetune

---

## Credits

- **VoxCPM model & training code** — [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) (Apache 2.0)
  `train_voxcpm_finetune.py` in this folder is vendored from that repo so the
  recipe stays self-contained — no separate clone required.
- **Base weights** — [openbmb/VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5)

---

## Hardware Requirements

| Setup | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 12 GB (batch_size=1, grad_accum=2) | 16–24 GB (batch_size=2) |
| RAM | 16 GB | 32 GB |
| Storage | 15 GB (base weights + checkpoints) | 30 GB |

**Training time** (100 clips, 2000 steps): ~1–2 hours on a single RTX 4090.
LoRA only updates a small adapter, so this is far lighter than full fine-tuning.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your dataset
#    See dataset/DATASET_FORMAT.md for the exact structure
cp your_wavs/*.wav dataset/audio/
#    then write dataset/train.jsonl — one {"audio": ..., "text": ...} per line

# 3. Configure
nano config.yaml     # set pretrained_path, num_iters, lora.r/alpha, save_path...

# 4. Train
python train.py --args.load=config.yaml

# 5. Monitor
tensorboard --logdir logs/lora_run1

# 6. Synthesize with your LoRA checkpoint
python infer.py \
    --lora_ckpt output/lora_run1/latest \
    --text "Hello, this is my fine-tuned voice." \
    --out result.wav
```

---

## Dataset Format

See **[dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md)** for the full spec.

**Quick summary:**
```
dataset/
├── train.jsonl        # {"audio": "...", "text": "..."} per line
├── val.jsonl          # optional
└── audio/
    ├── audio1.wav     # mono WAV, sample_rate matching config.yaml (default 44100 Hz)
    ├── audio2.wav
    └── ...
```

---

## Configuration Reference

Everything lives in **`config.yaml`**:

| Key | Default | Description |
|---|---|---|
| `pretrained_path` | `openbmb/VoxCPM1.5` | Base model — HF id or local snapshot path |
| `train_manifest` | `./dataset/train.jsonl` | Training manifest (JSONL) |
| `val_manifest` | `null` | Optional validation manifest |
| `sample_rate` | `44100` | Must match your audio files |
| `batch_size` | `2` | Per-step batch size |
| `grad_accum_steps` | `1` | Raise effective batch size without raising memory |
| `num_iters` / `max_steps` | `2000` | Total training steps |
| `learning_rate` | `0.0001` | LoRA adapter learning rate |
| `warmup_steps` | `100` | LR warmup before cosine decay |
| `save_interval` | `1000` | Checkpoint frequency (steps) |
| `save_path` | `./output/lora_run1` | Where checkpoints land |
| `tensorboard` | `./logs/lora_run1` | TensorBoard event dir |
| `max_batch_tokens` | `4096` | Drops samples too long for the batch (avoids OOM) |
| `lora.r` / `lora.alpha` | `16` / `32` | LoRA rank and scaling |
| `lora.enable_lm` / `enable_dit` | `true` / `true` | Which sub-modules get adapted |
| `lora.dropout` | `0.1` | Adapter dropout |

---

## Training Walkthrough

### Step 1 — Install

```bash
pip install -r requirements.txt
```

The base VoxCPM1.5 weights (~3 GB) download automatically from Hugging Face
on first run and are cached under `~/.cache/huggingface/hub/`.

### Step 2 — Prepare your dataset

See `dataset/DATASET_FORMAT.md`. In short:
- mono WAV at the `sample_rate` you set in `config.yaml`
- one `{"audio": "...", "text": "..."}` line per clip in `train.jsonl`
- transcripts must exactly match what's spoken — mismatches are the #1 cause
  of garbled LoRA output

### Step 3 — Configure

Open `config.yaml` and set, at minimum:
```yaml
pretrained_path: openbmb/VoxCPM1.5
train_manifest: ./dataset/train.jsonl
sample_rate: 44100
num_iters: 2000
save_path: ./output/lora_run1
```

**Memory tuning:**
- 24 GB GPU → `batch_size: 2`
- 12–16 GB GPU → `batch_size: 1`, `grad_accum_steps: 2`
- OOM during data loading → lower `max_batch_tokens`

### Step 4 — Train

```bash
python train.py --args.load=config.yaml
```

Loss prints every `log_interval` steps. With a small LoRA rank (16) and a
small dataset, 1500–2500 steps is usually enough to hear your target voice
and vocabulary come through.

### Step 5 — Monitor with TensorBoard

```bash
tensorboard --logdir logs/lora_run1
# open http://localhost:6006
```

### Step 6 — Pick a checkpoint

The trainer saves to `save_path/step_XXXXXXX/` every `save_interval` steps,
plus a `latest/` symlink-equivalent pointing at the most recent one. Each
checkpoint dir contains:
- `lora_weights.safetensors` — the adapter (use this for inference)
- `lora_config.json` — base model + LoRA hyperparameters (read automatically by `infer.py`)
- `optimizer.pth`, `scheduler.pth` — for resuming training

Don't judge from the earliest checkpoints alone — punctuation/symbol handling
can still be in flux and improve significantly with more steps (see the
example checkpoint's notes on `#` and `.`).

### Step 7 — Inference

```bash
python infer.py \
    --lora_ckpt output/lora_run1/latest \
    --text "Your custom text here, including tricky bits like # and 60.5%." \
    --out result.wav

# A/B compare against the un-adapted base model
python infer.py --lora_ckpt output/lora_run1/latest --text "..." --no_lora --out base.wav

# Voice cloning with a reference clip
python infer.py \
    --lora_ckpt output/lora_run1/latest \
    --text "This is voice cloning result." \
    --prompt_audio dataset/audio/audio1.wav \
    --prompt_text "Exact transcript of that reference clip" \
    --out clone.wav
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'voxcpm'`**
→ `pip install -r requirements.txt` (it installs `voxcpm` from GitHub).

**CUDA out of memory during training**
→ Lower `batch_size` to `1` and raise `grad_accum_steps` proportionally
(e.g. `batch_size: 1`, `grad_accum_steps: 2` keeps the effective batch the same).

**CUDA out of memory while loading data**
→ Lower `max_batch_tokens` (e.g. `2048`) so longer clips get filtered out.

**Model mispronounces symbols (`#`, `.`, `%`, etc.)**
→ Common in early checkpoints. Either keep training (LoRA can learn correct
pronunciation from your audio over more steps — this is exactly what happened
with `#` and `.` in the example checkpoint) or normalize those symbols to
spoken-word form in your transcripts (`#` → "hash", `%` → "percent").

**Resuming after an interrupted run**
→ Just re-run `python train.py --args.load=config.yaml` — it auto-loads
`optimizer.pth`/`scheduler.pth` from `save_path` and continues from the last
saved step. SIGINT/SIGTERM also trigger an immediate checkpoint save.

---

## Output Structure

```
output/lora_run1/
├── latest/
│   ├── lora_weights.safetensors
│   ├── lora_config.json
│   ├── optimizer.pth
│   └── scheduler.pth
├── step_0000000/
├── step_0001000/
└── step_0002000/
```
