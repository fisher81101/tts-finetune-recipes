#!/usr/bin/env python3
"""
Step 1: Prepare Dataset for Kokoro Fine-Tuning
===============================================
Reads data/train.csv, normalises text, runs G2P, and writes
training/train_list.txt and training/val_list.txt for StyleTTS2.

Usage:
    python scripts/01_prepare_dataset.py --config configs/config.yml

Output:
    training/train_list.txt   — StyleTTS2 format: wav_path|IPA|speaker_id
    training/val_list.txt
    training/normalization_samples.txt  — 20 random rows for spot-check
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import yaml


# ── Number-to-words (Indian numeral system: lakh, crore) ─────────────────────
# Used to expand amounts like "Rs.7,25,000" → "seven lakh twenty five thousand rupees"
# Extend or replace this block with your own locale's number words.

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _two(n: int) -> str:
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + (" " + _ONES[o] if o else "")


def _three(n: int) -> str:
    h, rest = divmod(n, 100)
    parts = []
    if h:
        parts.append(_ONES[h] + " hundred")
    if rest:
        if parts:
            parts.append("and")
        parts.append(_two(rest))
    return " ".join(parts)


def indian_number_words(n: int) -> str:
    if n == 0:
        return "zero"
    parts = []
    crore, rest = divmod(n, 10_000_000)
    if crore:
        parts.append(indian_number_words(crore) + " crore")
    lakh, rest = divmod(rest, 100_000)
    if lakh:
        parts.append(_two(lakh) + " lakh")
    thousand, rest = divmod(rest, 1000)
    if thousand:
        parts.append(_two(thousand) + " thousand")
    if rest:
        parts.append(_three(rest))
    return " ".join(parts)


# ── Text normalization ────────────────────────────────────────────────────────
# Tweak these regexes and functions for your own language/domain.

_RUPEES_RE      = re.compile(r"Rs\.?\s*(\d[\d,]*)", flags=re.IGNORECASE)
_INDIAN_GROUP_RE = re.compile(r"\b\d{1,2}(?:,\d{2})+,\d{3}\b")
_LONG_DIGITS_RE  = re.compile(r"\b\d{7,}\b")
_WORD_RE         = re.compile(r"\b[A-Za-z]+\b")
_LC_RUN_RE       = re.compile(r"[a-z]{2,}")


def _expand_rupees(m):
    digits = m.group(1).replace(",", "")
    try:
        return indian_number_words(int(digits)) + " rupees"
    except ValueError:
        return m.group(0)


def _expand_indian_group(m):
    digits = m.group(0).replace(",", "")
    try:
        return indian_number_words(int(digits))
    except ValueError:
        return m.group(0)


def _spell_long_digits(m):
    return " ".join(m.group(0))


def _maybe_spell_acronym(m):
    """Acronyms (MMS, IST, CEO) → letter-by-letter.  Proper names (Naren, iPhone) pass through."""
    word = m.group(0)
    if sum(1 for c in word if c.isupper()) < 2:
        return word
    if _LC_RUN_RE.search(word):
        return word
    return " ".join(c.upper() for c in word)


def normalize_text(text: str) -> str:
    """Apply text normalization to a transcript string.

    Extend this for your own language or domain:
    - Add new regex patterns for domain-specific text (dates, units, abbreviations)
    - Replace the rupee expander with your currency
    - Swap the acronym spellout for your language's conventions
    """
    text = text.replace("@", " at ").replace("&", " and ")
    text = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))
    # slash between letters → space (AI/ML → AI ML, CI/CD → CI CD)
    text = re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", " ", text)
    text = _RUPEES_RE.sub(_expand_rupees, text)
    text = _LONG_DIGITS_RE.sub(_spell_long_digits, text)
    text = _INDIAN_GROUP_RE.sub(_expand_indian_group, text)
    text = _WORD_RE.sub(_maybe_spell_acronym, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    ds = cfg["dataset"]
    g2p_cfg = cfg.get("g2p", {})
    fw = cfg.get("framework", {})

    root = Path(config_path).parent.parent
    data_csv = root / ds["csv_path"]
    wavs_dir = root / ds["wavs_dir"]
    val_ratio = float(ds.get("val_ratio", 0.05))
    seed = int(ds.get("seed", 42))

    def _resolve(raw: str, fallback: Path) -> Path:
        p = Path(raw) if raw else None
        if p:
            return p if p.is_absolute() else (root / p).resolve()
        return fallback

    # training_dir: where train_list.txt and val_list.txt are written.
    # StyleTTS2 scripts read them from this location.
    training_dir = _resolve(
        fw.get("training_dir", ""),
        _resolve(fw.get("kokoro_deutsch_dir", ""), Path()) / "training"
        if fw.get("kokoro_deutsch_dir") else root / "framework" / "training"
    )
    training_dir.mkdir(parents=True, exist_ok=True)

    train_list = training_dir / "train_list.txt"
    val_list   = training_dir / "val_list.txt"
    samples_file = training_dir / "normalization_samples.txt"

    language = g2p_cfg.get("language", "en-gb")

    # Import G2P from this recipe's scripts/
    scripts_dir = Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from g2p_helper import g2p_with_lexicon

    # Load CSV
    with open(data_csv, encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip()]
    rows = list(csv.DictReader(lines))
    print(f"Loaded {len(rows)} rows from {data_csv}")

    entries = []
    missing_wav = 0
    g2p_fail = 0
    empty_ipa = 0

    for row in rows:
        wid = row["id"]
        wav_path = wavs_dir / f"{wid}.wav"
        if not wav_path.exists():
            missing_wav += 1
            continue
        text = row["text"]
        norm = normalize_text(text)
        try:
            phonemes = g2p_with_lexicon(norm, language=language).strip()
        except Exception as e:
            print(f"  G2P fail id={wid}: {e}")
            g2p_fail += 1
            continue
        if len(phonemes) < 5:
            empty_ipa += 1
            continue
        entries.append({
            "filename": str(wav_path.resolve()),
            "ipa": phonemes,
            "speaker": "0",
            "orig": text,
            "norm": norm,
        })

    print(f"Valid entries : {len(entries):,}")
    if missing_wav:
        print(f"Missing WAV   : {missing_wav}")
    if empty_ipa:
        print(f"Empty phonemes: {empty_ipa}")
    if g2p_fail:
        print(f"G2P failures  : {g2p_fail}")

    if not entries:
        print("[ERROR] No valid entries. Check csv_path and wavs_dir in config.yml")
        sys.exit(1)

    rng = random.Random(seed)
    rng.shuffle(entries)
    n_val = max(1, int(len(entries) * val_ratio))
    val   = entries[:n_val]
    train = entries[n_val:]
    print(f"Split: train={len(train)}, val={len(val)}")

    for path, items in [(train_list, train), (val_list, val)]:
        with open(path, "w") as f:
            for e in items:
                f.write(f"{e['filename']}|{e['ipa']}|{e['speaker']}\n")
        print(f"Wrote {path}  ({len(items)} lines)")

    # Dump 20 random samples for manual spot-check
    rng2 = random.Random(0)
    picks = rng2.sample(entries, min(20, len(entries)))
    with open(samples_file, "w") as f:
        f.write("Normalization + G2P samples (20 random training rows)\n")
        f.write("=" * 80 + "\n\n")
        for e in picks:
            f.write(f"ORIG: {e['orig']}\n")
            f.write(f"NORM: {e['norm']}\n")
            f.write(f"IPA : {e['ipa']}\n\n")
    print(f"Wrote {samples_file}")
    print("\nDone. Check normalization_samples.txt for a quick sanity check.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/config.yml", help="Path to config.yml")
    args = ap.parse_args()
    main(args.config)
