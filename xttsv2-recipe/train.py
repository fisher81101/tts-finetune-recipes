#!/usr/bin/env python3
"""
XTTS v2 Fine-Tuning — Training Script
======================================
All parameters are read from environment variables so you can tune them
in config.env without touching this file.

Run:
    source config.env && python train.py            # single GPU
    source config.env && python train.py --multi    # print multi-GPU command

Original training recipe by Coqui AI (https://github.com/coqui-ai/TTS)
Adapted for generic fine-tuning use.
"""
import argparse
import os
import sys

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# ── Logging parameters ────────────────────────────────────────────────────────
RUN_NAME = "GPT_XTTS_v2.0_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None


# ── Env-var helpers (keep in sync with config.env keys) ──────────────────────

def _get_env_str(name, default):
    v = os.getenv(name)
    return v if v else default

def _get_env_int(name, default):
    v = os.getenv(name)
    return int(v) if v else default

def _get_env_float(name, default):
    v = os.getenv(name)
    return float(v) if v else default

def _get_env_bool(name, default):
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "y", "on"}

def _get_env_int_list(name, default):
    v = os.getenv(name)
    if not v:
        return default
    tokens = [t.strip() for t in v.split(",") if t.strip()]
    return [int(t) for t in tokens] if tokens else default


# ── Read all parameters from environment ─────────────────────────────────────

DATASET_PATH     = _get_env_str("DATASET_PATH",     "./dataset")
METADATA_FILE    = _get_env_str("METADATA_FILE",    os.path.join(DATASET_PATH, "metadata.csv"))
SPEAKER_REFERENCE = _get_env_str("SPEAKER_REFERENCE", os.path.join(DATASET_PATH, "reference.wav"))
OUT_PATH          = _get_env_str("OUT_PATH",          "./output")
LANGUAGE          = _get_env_str("LANGUAGE",          "en")

BATCH_SIZE        = _get_env_int("XTTS_BATCH_SIZE",   4)
GRAD_ACUMM_STEPS  = _get_env_int("XTTS_GRAD_ACCUM_STEPS", 16)
_eval_bs_raw      = os.getenv("XTTS_EVAL_BATCH_SIZE")
EVAL_BATCH_SIZE   = int(_eval_bs_raw) if _eval_bs_raw else BATCH_SIZE

TEST_TEXT_1 = _get_env_str("XTTS_TEST_TEXT_1", "The quick brown fox jumps over the lazy dog.")
TEST_TEXT_2 = _get_env_str("XTTS_TEST_TEXT_2", "Hello, this is a fine-tuned XTTS voice speaking.")

# For multi-gpu: False keeps WD on all params (required for DDP)
OPTIMIZER_WD_ONLY_ON_WEIGHTS = False


def _check_inputs():
    errors = []
    if not os.path.isdir(DATASET_PATH):
        errors.append(f"DATASET_PATH not found: {DATASET_PATH}")
    if not os.path.isfile(METADATA_FILE):
        errors.append(f"METADATA_FILE not found: {METADATA_FILE}")
    if not os.path.isfile(SPEAKER_REFERENCE):
        errors.append(f"SPEAKER_REFERENCE not found: {SPEAKER_REFERENCE}\n"
                      f"  Place a clean 5-10s WAV of your speaker here (24 kHz).")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        sys.exit(1)


def _print_multi_gpu_command():
    gpus = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    omp  = os.getenv("OMP_NUM_THREADS", "2")
    print()
    print("Multi-GPU (DDP) launch command:")
    print(f"  OMP_NUM_THREADS={omp} CUDA_VISIBLE_DEVICES={gpus} \\")
    print(f"    python3 -m trainer.distribute --script train.py")
    print()
    print("Re-run without --multi to train on a single GPU.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", action="store_true",
                    help="Print the multi-GPU DDP command and exit.")
    args = ap.parse_args()

    if args.multi:
        _print_multi_gpu_command()
        sys.exit(0)

    _check_inputs()

    # ── XTTS v2 base model files (auto-downloaded on first run) ──────────────
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    DVAE_LINK       = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK   = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    TOKENIZER_LINK  = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CKPT_LINK  = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
    MEL_NORM_FILE   = os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth")
    TOKENIZER_FILE  = os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json")
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "model.pth")

    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print("[info] Downloading DVAE + mel stats...")
        ModelManager._download_model_files(
            [MEL_NORM_LINK, DVAE_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        print("[info] Downloading XTTS v2 base checkpoint...")
        ModelManager._download_model_files(
            [TOKENIZER_LINK, XTTS_CKPT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    # Uses the "thorsten" formatter which expects: id|text  (2 columns, no header)
    # If your metadata has 3 columns (id|text|speaker) use formatter="ljspeech"
    config_dataset = BaseDatasetConfig(
        formatter="thorsten",
        dataset_name="custom",
        path=DATASET_PATH + "/",
        meta_file_train=METADATA_FILE,
        language=LANGUAGE,
    )
    DATASETS_CONFIG_LIST = [config_dataset]

    # ── Model args ────────────────────────────────────────────────────────────
    model_args = GPTArgs(
        max_conditioning_length=132300,   # 6 secs at 22050 Hz
        min_conditioning_length=66150,    # 3 secs
        debug_loading_failures=False,
        max_wav_length=_get_env_int("XTTS_MAX_WAV_LENGTH", 350000),
        max_text_length=500,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        epochs=_get_env_int("XTTS_EPOCHS", 30),
        run_description="XTTS v2 fine-tuning",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=_get_env_int("XTTS_BATCH_GROUP_SIZE", 32),
        eval_batch_size=EVAL_BATCH_SIZE,
        num_loader_workers=_get_env_int("XTTS_NUM_LOADER_WORKERS", 8),
        eval_split_max_size=_get_env_int("XTTS_EVAL_SPLIT_MAX_SIZE", 80),
        eval_split_size=_get_env_float("XTTS_EVAL_SPLIT_SIZE", 0.1),
        print_step=_get_env_int("XTTS_PRINT_STEP", 50),
        plot_step=_get_env_int("XTTS_PLOT_STEP", 100),
        log_model_step=_get_env_int("XTTS_LOG_MODEL_STEP", 400),
        save_step=_get_env_int("XTTS_SAVE_STEP", 200),
        save_n_checkpoints=_get_env_int("XTTS_SAVE_N_CHECKPOINTS", 1),
        save_checkpoints=True,
        print_eval=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={
            "betas": [0.9, 0.96],
            "eps": 1e-8,
            "weight_decay": _get_env_float("XTTS_WEIGHT_DECAY", 1e-2),
        },
        lr=_get_env_float("XTTS_LR", 5e-6),
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": _get_env_int_list("XTTS_LR_MILESTONES", [900, 1400, 1850]),
            "gamma": _get_env_float("XTTS_LR_GAMMA", 0.5),
            "last_epoch": -1,
        },
        test_sentences=[
            {"text": TEST_TEXT_1, "speaker_wav": [SPEAKER_REFERENCE], "language": LANGUAGE},
            {"text": TEST_TEXT_2, "speaker_wav": [SPEAKER_REFERENCE], "language": LANGUAGE},
        ],
    )

    model = GPTTrainer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=_get_env_bool("XTTS_START_WITH_EVAL", True),
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
