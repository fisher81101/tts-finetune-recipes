# Dataset Format

VoxCPM's LoRA trainer reads a JSONL manifest — one JSON object per line,
each pointing at an audio file and its transcript:

```jsonl
{"audio": "./dataset/audio/audio1.wav", "text": "The training session lasted for one hour, and after an hour we reviewed the results together."}
{"audio": "./dataset/audio/audio2.wav", "text": "The sensor updates several times per second, and the readings per second must remain stable during testing."}
```

## Layout

```
dataset/
├── train.jsonl        # one {"audio": ..., "text": ...} per line
├── val.jsonl          # optional — same format, used if val_manifest is set
└── audio/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

## Audio requirements

- Mono WAV
- Sample rate matching `sample_rate` in `config.yaml` (default `44100`)
- A few seconds to ~20s per clip works well; trim silence at the edges
- `audio` paths in the manifest can be relative (resolved from the working
  directory you launch `train.py` from) or absolute

## Transcript tips

- Write numbers, symbols, and abbreviations the way you want them *spoken*
  if you need precise control (e.g. spell out `#` as "hash" or "number" if
  the base model mispronounces it) — or leave them as-is and let LoRA learn
  the mapping from your audio, the way the example checkpoint did with `#`
  and `.` in technical sentences.
- Keep transcripts an exact match of what's spoken in the clip — mismatches
  are the #1 cause of garbled LoRA output.

## Quick sanity check

```bash
python -c "
import json
n = 0
for line in open('dataset/train.jsonl', encoding='utf-8'):
    d = json.loads(line)
    assert 'audio' in d and 'text' in d
    n += 1
print(f'{n} entries OK')
"
```
