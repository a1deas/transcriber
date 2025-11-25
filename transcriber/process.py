# transcriber/transcriber/srt_writer.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, List, Dict

from faster_whisper import WhisperModel
from rich.console import Console

from .ffmpeg_utils import extract_wav
from .srt_writer import write_srt

console = Console()

TaskType = Literal["transcribe", "translate"]

# Config model function to re-use model afterwards.
def load_model(
    model_name: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16",
) -> WhisperModel:
    console.log(f"[bold cyan] Loading faster-whisper model[/] {model_name!r} on {device}, type = {compute_type}")
    
    model = WhisperModel(
        model_name,
        device = device,
        compute_type = compute_type,
    )
    return model

# High-level function: extracts audio -> transcribes -> writes .srt and .json
def transcribe(
        input_path: Path,
        output_srt: Path,
        output_json: Path, 
        model_name: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        task: TaskType = "transcribe",
) -> dict:
    input_path = Path(input_path)
    output_srt = Path(output_srt)
    output_json = Path(output_json)

    workdir = output_json.parent
    workdir.mkdir(parents = True, exist_ok = True)
    wav_path = workdir / (input_path.stem + ".normalized.wav")

    console.log(f"[green] Extracting audio[/] from {input_path} -> {wav_path}")
    extract_wav(input_path, wav_path)

    try: 
        model = load_model(
            model_name = model_name,
            device = device,
            compute_type = compute_type,
        )
    except Exception as e:
        console.log(f"[yellow] Failed to load model on {device}: {e}. Falling back to CPU.[/]")
        model = load_model(
            model_name = model_name,
            device = "cpu",
            compute_type = "int8",
        )

    console.log(f"[green] Transcribing[/] {wav_path} (task = {task}, language = {language or 'auto'})")

    segments_iter, info = model.transcribe(
        str(wav_path),
        task = task,
        language = language, # None -> auto
    )

    segments: List[Dict] = []
    for segment in segments_iter:
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text,
                "lang": info.language,

            }
        )

    console.log(f"[blue] Got {len(segments)} segments, language = {info.language}, duration = {info.duration:.1f}s[/]")

    # SRT 
    write_srt(segments, output_srt)
    console.log(f"[green] Written[/] SRT -> {output_srt}")

    # JSON
    import json
    output_json.write_text(
        json.dumps(
            {
                "language": info.language,
                "duration": info.duration,
                "segments": segments,
            },
            ensure_ascii = False,
            indent = 2,
        ),
        encoding = "utf-8"
    )

    console.log(f"[green] Written[/] JSON -> {output_json}")

    return {
        "language": info.language,
        "duration": info.duration,
        "segments_count": len(segments),
        "srt": str(output_srt),
        "json": str(output_json),
    }