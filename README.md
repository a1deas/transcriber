# Transcriber
A lightweight, GPU-accelerated CLI tool for speech-to-text transcription powered by Faster-Whisper + FFmpeg.

Designed as a standalone utility and as a reusable module for larger pipelines(video automation tools, ml workflows, etc...)

## Features
- Extracts and normalizes audio from video/audio files (.mp4, .mkv, .mp3, .wav, etc.)
- Converts audio to WAV 16 kHz mono (FFmpeg)
- Fast transcription via Faster-Whisper (CUDA / CPU)
- Automatic language detection or manual override
- Two modes:
  - transcribe — speech → text
  - translate — speech → translated text
- Generates:
  - output.srt — standard SRT subtitles
  - output.json — structured segment metadata
- Automatic fallback to CPU if GPU is unavailable

## Installation (Poetry)
```text
git clone https://github.com/a1deas/transcriber.git
cd transcriber
poetry install
```

## Usage
Basic transcription
```text
poetry run transcriber input.mp4 \
  --srt outputs/subs.srt \
  --json outputs/subs.json
```
| Option             | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| `input_path`       | Path to input audio/video file (positional)                   |
| `--srt`            | Output `.srt` file path                                       |
| `--json`           | Output `.json` file path                                      |
| `--model`, `-m`    | Whisper model (`small`, `medium`, `large`, or local path)     |
| `--device`, `-d`   | `cuda` or `cpu`                                               |
| `--compute-type`   | Precision: `float16`, `float32`, `int8`, `int8_float16`, etc. |
| `--language`, `-l` | Force language (`en`, `ru`, `fr`, etc.). Default: auto        |
| `--task`, `-t`     | `transcribe` or `translate`                                   |

## Integration(Python API)
```text
from transcriber.process import transcribe

result = transcribe(
    input_path="input.mp4",
    output_srt="output.srt",
    output_json="output.json",
    model_name="medium",
    device="cuda",
)

```

## License
Licensed under the MIT License.
Free to use in commercial and private projects.


## Notes
- CUDA is used automatically if available.
  - CPU fallback uses int8 precision for performance.
-  Models are cached automatically.
