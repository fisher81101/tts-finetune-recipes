# Dataset Format Guide — Kokoro Fine-Tuning

## Directory Layout

```
data/
├── train.csv          ← transcript file
└── wavs/
    ├── 0001.wav
    ├── 0002.wav
    └── ...
```

## train.csv

Three columns: `id,target_audio,text` — with a header row.

```
id,target_audio,text
0001,/path/to/0001.wav,The quick brown fox jumps over the lazy dog.
0002,/path/to/0002.wav,She sells seashells by the seashore.
0003,/path/to/0003.wav,How much wood would a woodchuck chuck.
```

- `id` — must match the WAV filename (without `.wav`)
- `target_audio` — can be any path or even a placeholder; `01_prepare_dataset.py`
  derives the actual WAV path from `id` and ignores this column
- `text` — the transcript (UTF-8)
- The header row (`id,target_audio,text`) is required

## Audio Specs

| Property | Requirement |
|---|---|
| Format | WAV (PCM) |
| Channels | Mono |
| Sample rate | **24000 Hz** |
| Duration | 3 – 25 seconds per clip |
| Amplitude | Normalized (no clipping) |

Clips shorter than ~3 seconds have fewer mel frames than the training window
(`max_len=180` frames = 2.25 sec). Very short clips are fine but contribute
less to style learning.

## Dataset Size Recommendations

| Dataset size | Expected quality |
|---|---|
| 200–500 clips | Moderate voice adaptation |
| 500–1000 clips | Good quality, accent/style transfer |
| 1000+ clips | Best results |

StyleTTS2 is sample-efficient — 500 clips of 10-25 sec each (~3-4 hours) typically
produces a convincing voice clone.

## Converting Audio

Convert your audio to 24 kHz mono WAV with ffmpeg:

```bash
for f in raw_audio/*.mp3; do
    id=$(basename "$f" .mp3)
    ffmpeg -i "$f" -ar 24000 -ac 1 data/wavs/${id}.wav -y
done
```

## Custom Pronunciation (Lexicon)

If your dataset contains proper nouns, brand names, or technical terms that the
G2P engine mispronounces, add them to `scripts/lexicon.json`:

```json
{
  "MyBrand": "mˈI bɹænd",
  "YourCity": "jˈʊɹ sˈɪti"
}
```

Run a quick G2P check before training:

```python
from scripts.g2p_helper import g2p_with_lexicon
print(g2p_with_lexicon("MyBrand YourCity"))
```

Values must use phonemes from Kokoro's 178-phoneme vocab. Kokoro shortcuts:
`A`=/eɪ/ `I`=/aɪ/ `O`=/oʊ/ `W`=/aʊ/ `Y`=/ɔɪ`. Do NOT use length marks `ː`.
