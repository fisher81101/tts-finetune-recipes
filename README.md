# TTS Fine-Tuning Recipes

Self-contained recipes for fine-tuning text-to-speech models on custom speaker data.
Drop in your dataset, tweak one config file, and run — no cloning required.

---

## Recipes

### 🗣 XTTSv2 (Coqui)
Fine-tune **XTTS v2** — a 518M-parameter GPT-based multilingual TTS model with voice cloning.

| | |
|---|---|
| 📁 Recipe | [xttsv2-recipe/](./xttsv2-recipe/) |
| 🤗 Example checkpoint | [![HuggingFace](https://img.shields.io/badge/🤗-xtts--v2--indian--en-ff6b00?style=flat-square)](https://huggingface.co/jeevav62/xtts-v2-indian-en) |
| Hardware | 16–24 GB GPU VRAM |
| Training time | ~2–3 hours (1000 clips, 30 epochs, RTX 4090) |

### 🗣 Kokoro-82M (hexgrad)
Fine-tune **Kokoro-82M** — a lightweight StyleTTS2-based TTS model with fast inference.

| | |
|---|---|
| 📁 Recipe | [kokoro-recipe/](./kokoro-recipe/) |
| 🤗 Example checkpoint | [![HuggingFace](https://img.shields.io/badge/🤗-kokoro--82m--indian--en-ff6b00?style=flat-square)](https://huggingface.co/jeevav62/kokoro-82m-indian-en) |
| Hardware | 12–24 GB GPU VRAM |
| Training time | ~75 min Stage 1 + ~3 hours Stage 2 with GAN (1000 clips, RTX 4090) |

### 🗣 VoxCPM 1.5 (OpenBMB) — LoRA
Fine-tune **VoxCPM 1.5** — a tokenizer-free, diffusion-based TTS model (MiniCPM-4 backbone) — with lightweight LoRA adapters instead of full fine-tuning.

| | |
|---|---|
| 📁 Recipe | [voxcpm-recipe/](./voxcpm-recipe/) |
| 🤗 Example checkpoint | [![HuggingFace](https://img.shields.io/badge/🤗-voxcpm--lora--finetune-ff6b00?style=flat-square)](https://huggingface.co/jeevav62/voxcpm-lora-finetune) |
| Hardware | 12–24 GB GPU VRAM |
| Training time | ~1–2 hours (100 clips, 2000 steps, RTX 4090) |

---

## How to use

Each recipe is self-contained — just go into the folder and follow its README:

```bash
# XTTSv2
cd xttsv2-recipe
pip install -r requirements.txt
# add your dataset → source config.env && python train.py

# Kokoro
cd kokoro-recipe
pip install -r requirements.txt
# add your dataset → python scripts/01_prepare_dataset.py && python scripts/02_train.py

# VoxCPM 1.5 (LoRA)
cd voxcpm-recipe
pip install -r requirements.txt
# add your dataset → python train.py --args.load=config.yaml
```

---

## Credits

- **[Coqui TTS / XTTS v2](https://github.com/coqui-ai/TTS)** — MIT License
- **[hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** — Apache 2.0
- **[yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)** — MIT License
- **[semidark/kokoro-deutsch](https://github.com/semidark/kokoro-deutsch)** — Apache 2.0
- **[OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)** — Apache 2.0
