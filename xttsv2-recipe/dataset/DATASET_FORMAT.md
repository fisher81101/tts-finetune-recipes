# Dataset Format Guide — XTTS v2

## Directory Layout

```
dataset/
├── metadata.csv       ← transcript file
├── reference.wav      ← clean speaker reference (5-10 sec, 24 kHz)
└── wavs/
    ├── 0001.wav
    ├── 0002.wav
    └── ...
```

## metadata.csv

Pipe-delimited, **no header line**, two columns: `id|text`

```
0001|The quick brown fox jumps over the lazy dog.
0002|She sells seashells by the seashore.
0003|How much wood would a woodchuck chuck.
```

- `id` must match the WAV filename (without `.wav`).
- Do **not** add a header row — the parser treats every line as data.
- Use UTF-8 encoding.

## Audio Specs

| Property | Requirement |
|---|---|
| Format | WAV (PCM) |
| Channels | Mono |
| Sample rate | **22050 Hz** |
| Duration | 1 – 24 seconds per clip |
| Amplitude | Normalized (no clipping) |

The reference wav (`reference.wav`) should be:
- The **same speaker** as the training data
- 5–10 seconds long
- Clean (minimal background noise)
- **24 kHz** (XTTS outputs at 24 kHz)

## Dataset Size Recommendations

| Dataset size | Expected quality |
|---|---|
| 100–300 clips | Minimal adaptation, accent/style transfer |
| 500–1000 clips | Good voice cloning quality |
| 1000+ clips | Best results, especially for domain-specific vocab |

## Converting Audio

If your audio is not 22050 Hz mono, convert it with ffmpeg:

```bash
# Batch convert a directory of MP3 files to 22050 Hz mono WAV
for f in raw_audio/*.mp3; do
    id=$(basename "$f" .mp3)
    ffmpeg -i "$f" -ar 22050 -ac 1 dataset/wavs/${id}.wav -y
done
```

## Naming Convention

IDs must be consistent between `metadata.csv` and the `wavs/` directory.
You can use any naming scheme (numeric, alphanumeric, etc.) as long as
`wavs/{id}.wav` exists for every `id` in `metadata.csv`.
