# XTTS v2 Fine-Tuning Recipe

Fine-tune Coqui's XTTS v2 (a 518M-parameter GPT-based TTS model) on your own speaker data.
The result is a model that speaks in your target voice and handles your domain vocabulary well.

> **Example checkpoint trained with this recipe:**
> [jeevav62/xtts-v2-indian-en](https://huggingface.co/jeevav62/xtts-v2-indian-en)
> — XTTS v2 fine-tuned on Indian-English (male voice, 1058 clips, step 11074)

---

## Credits

- **XTTS v2 model** — [Coqui AI](https://github.com/coqui-ai/TTS) (MIT License)
  The original TTS library and GPT-XTTS architecture.
- **Training recipe** — Derived from `TTS/recipes/ljspeech/xtts_v2/train_gpt_xtts.py`
  by the Coqui contributors.
- **Thorsten dataset format** — Named after [Thorsten Müller](https://github.com/thorstenMueller),
  whose open-source voice dataset established the `id|text` two-column convention.

---

## Hardware Requirements

| Setup | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 16 GB (batch_size=2) | 24 GB (batch_size=4) |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB (checkpoints + base model) | 50 GB |

**Training time** (1000 clips, 30 epochs):
- 1× RTX 4090 (24 GB): ~2–3 hours
- 2× RTX 3090 (24 GB each): ~1–1.5 hours

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your dataset
#    See dataset/DATASET_FORMAT.md for the exact structure
cp your_wavs/*.wav dataset/wavs/
cp your_metadata.csv dataset/metadata.csv
cp your_speaker_ref.wav dataset/reference.wav

# 3. Configure
nano config.env      # set LANGUAGE, tweak BATCH_SIZE, EPOCHS, etc.

# 4. Train (single GPU)
source config.env && python train.py

# 5. Train (multi-GPU — prints the DDP command)
source config.env && python train.py --multi

# 6. Monitor
tensorboard --logdir output/

# 7. Synthesize with your trained model
python infer.py \
    --text "Hello, this is my fine-tuned voice." \
    --speaker-wav dataset/reference.wav \
    --checkpoint output/<run-dir>/best_model.pth \
    --config    output/<run-dir>/config.json \
    --out output.wav
```

---

## Dataset Format

See **[dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md)** for the full spec.

**Quick summary:**
```
dataset/
├── metadata.csv       # id|text (pipe-delimited, no header)
├── reference.wav      # clean speaker reference, 24 kHz, 5-10 sec
└── wavs/
    ├── 0001.wav       # 22050 Hz mono WAV, 1-24 sec
    ├── 0002.wav
    └── ...
```

---

## Configuration Reference

All parameters live in **`config.env`**. Source it before running:
```bash
source config.env && python train.py
```

| Variable | Default | Description |
|---|---|---|
| `DATASET_PATH` | `./dataset` | Root of your dataset directory |
| `METADATA_FILE` | `./dataset/metadata.csv` | Pipe-delimited id\|text file |
| `SPEAKER_REFERENCE` | `./dataset/reference.wav` | Speaker reference WAV (24 kHz) |
| `OUT_PATH` | `./output` | Where checkpoints are saved |
| `CUDA_VISIBLE_DEVICES` | `0` | GPUs to use (`0`, `0,1`, etc.) |
| `LANGUAGE` | `en` | Language code |
| `XTTS_EPOCHS` | `30` | Training epochs |
| `XTTS_BATCH_SIZE` | `4` | Per-GPU batch size |
| `XTTS_GRAD_ACCUM_STEPS` | `16` | Gradient accumulation steps |
| `XTTS_LR` | `5e-6` | Learning rate |
| `XTTS_MAX_WAV_LENGTH` | `350000` | Max clip length in samples (~16s) |
| `XTTS_EVAL_SPLIT_SIZE` | `0.1` | Validation fraction (10%) |
| `XTTS_SAVE_STEP` | `200` | Save checkpoint every N steps |
| `XTTS_TEST_TEXT_1` | _(generic)_ | Sentence synthesized during eval |
| `XTTS_TEST_TEXT_2` | _(generic)_ | Second eval sentence |

---

## Training Walkthrough

### Step 1 — Install

```bash
pip install -r requirements.txt
```

The base XTTS v2 model weights (~2 GB) are downloaded automatically on the first training run.

### Step 2 — Prepare your dataset

See `dataset/DATASET_FORMAT.md`. Ensure:
- Audio is 22050 Hz mono WAV
- `metadata.csv` has `id|text` format (no header, pipe-delimited)
- `reference.wav` is a clean 5-10 sec sample of your speaker at 24 kHz

### Step 3 — Configure

Open `config.env` and at minimum set:
```bash
DATASET_PATH=./dataset
SPEAKER_REFERENCE=./dataset/reference.wav
LANGUAGE=en
XTTS_EPOCHS=30
```

**Memory tuning:**
- 24 GB GPU → `XTTS_BATCH_SIZE=4`
- 16 GB GPU → `XTTS_BATCH_SIZE=2`, `XTTS_GRAD_ACCUM_STEPS=32`
- OOM during training → reduce `XTTS_MAX_WAV_LENGTH` to `255000` or lower

### Step 4 — Train

**Single GPU:**
```bash
source config.env && python train.py
```

**Two GPUs (DDP):**
```bash
source config.env && python train.py --multi
# Copy the printed command and run it
```

Training prints loss every 50 steps. Eval loss should decrease steadily.
A rough benchmark: eval loss dropping from ~3.9 → ~2.7 over 30 epochs is healthy.

### Step 5 — Monitor with TensorBoard

```bash
tensorboard --logdir output/
# Open http://localhost:6006 in your browser
```

Watch `eval/loss` — the checkpoint saved at the lowest eval loss is your `best_model.pth`.

### Step 6 — Pick the best checkpoint

The trainer auto-saves `best_model.pth` whenever eval loss improves.
You can also listen to the test-sentence WAVs written under `output/<run>/` at each eval step.

### Step 7 — Inference

```bash
python infer.py \
    --text "Your custom text here." \
    --speaker-wav dataset/reference.wav \
    --checkpoint output/<run-dir>/best_model.pth \
    --config    output/<run-dir>/config.json \
    --out result.wav
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'TTS'`**
→ Run `pip install -r requirements.txt` first.

**CUDA out of memory during training**
→ Reduce `XTTS_BATCH_SIZE` to 2 or 1 in `config.env`, and increase `XTTS_GRAD_ACCUM_STEPS` proportionally so the product stays ≥ 252.

**CUDA out of memory during data loading**
→ Reduce `XTTS_MAX_WAV_LENGTH` to `255000` (11.6 sec) or `220000` (10 sec).

**`formatter="thorsten"` error / wrong column count**
→ Your `metadata.csv` must be pipe-delimited with exactly 2 columns (`id|text`). If you have 3 columns (`id|text|speaker`) use the `ljspeech` formatter — edit `train.py` line that says `formatter="thorsten"` to `formatter="ljspeech"`.

**UNK token warnings for special characters (`<`, `@`, `%`)**
→ Normal — the tokenizer drops these symbols. Replace them in your transcripts with their spoken equivalents before training (e.g., `@` → `at`, `%` → `percent`).

**`RecursionError` during training**
→ Your audio clips are too short relative to `min_conditioning_length`. Set `XTTS_MAX_WAV_LENGTH` to at least `130000` (6 sec) so the conditioning window fits. If clips are genuinely short, remove clips under 3 seconds from your dataset.

**Multi-GPU training fails with `NCCL error`**
→ Ensure both GPUs are on the same node and NCCL is installed (`pip install torch` includes it). Double-check `CUDA_VISIBLE_DEVICES` in `config.env`.

---

## Output Structure

```
output/
└── GPT_XTTS_v2.0_FT-<date>/
    ├── best_model.pth          ← best checkpoint (use this for inference)
    ├── best_model_NNNN.pth     ← same, named with step number
    ├── checkpoint_NNNN.pth     ← periodic checkpoint
    ├── config.json             ← training config (needed for inference)
    ├── trainer_0_log.txt       ← training log
    ├── events.out.tfevents.*   ← TensorBoard events
    └── XTTS_v2.0_original_model_files/
        ├── model.pth           ← base XTTS v2 weights (downloaded once)
        ├── vocab.json
        ├── dvae.pth
        └── mel_stats.pth
```
