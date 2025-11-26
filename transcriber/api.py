# transcriber/transcriber/api.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any

from faster_whisper import WhisperModel
from rich.console import Console

from .ffmpeg_utils import extract_wav
from .srt_writer import write_srt

console = Console()

TaskType = Literal["transcribe", "translate"]
QualityType = Literal["fast", "balanced", "quality"]

@dataclass
class Segment: 
    start: float
    end: float
    text: str
    lang: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None

@dataclass
class TranscriptionResult: 
    language: str 
    duration: float
    segments: List[Segment]

# simple model module-level cache
_MODEL: Optional[WhisperModel] = None
_MODEL_KEY: Optional[tuple[str, str, str]] = None

# Config model function to re-use model afterwards.
def _get_model(
    model_name: str,
    device: str,
    compute_type: str,
) -> WhisperModel:
    global _MODEL, _MODEL_KEY

    key = (model_name, device, compute_type)

    if _MODEL is not None and _MODEL_KEY == key: 
        return _MODEL

    console.log(f"[bold cyan] Loading faster-whisper model[/] {model_name!r} on [magenta]{device}, type = {compute_type}")
    
    _MODEL = WhisperModel(
        model_name,
        device = device,
        compute_type = compute_type,
        num_workers = 4,
    )
    _MODEL_KEY = key
    return _MODEL

def _resolve_quality(
        quality: str,
) -> Dict[str, Any]:
    q = quality.lower()
    if q == "fast":
        return dict(
            beam_size = 1,
            vad_filter = False,
            word_timestamps = False,
        ) 
    elif q == "quality":
        return dict(
            beam_size = 5,
            vad_filter = True,
            word_timestamps = True,
        )
    else: # balanced / default
        return dict(
            beam_size = 3, 
            vad_filter = True, 
            word_timestamps = False,
        )

# High-level function: extracts audio -> transcribes -> writes .srt and .json
def transcribe_file(
        input_path: Path,
        model_name: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        task: TaskType = "transcribe",
        quality: QualityType = "balanced",
        output_srt: Optional[Path] = None,
        output_json: Optional[Path] = None, 
) -> dict:
    input_path = Path(input_path)

    if output_json is not None: 
        workdir = Path(output_json).parent
    else: 
        workdir = input_path.parent / "transcriber_cache"

    workdir.mkdir(parents = True, exist_ok = True)
    wav_path = workdir / (input_path.stem + ".normalized.wav")

    console.log(f"[green] Extracting audio[/] from {input_path} -> {wav_path}")
    extract_wav(input_path, wav_path)

    try: 
        model = _get_model(
            model_name = model_name,
            device = device,
            compute_type = compute_type,
        )
    except Exception as e:
        console.log(f"[yellow] Failed to load model on {device}: {e}. Falling back to CPU.[/]")
        model = _get_model(
            model_name = model_name,
            device = "cpu",
            compute_type = "int8",
        )

    q_params = _resolve_quality(quality)

    console.log(f"[green] Transcribing[/] {wav_path} "
                f"(task = {task}, language = {language or 'auto'}, quality = {quality})")

    segments_iter, info = model.transcribe(
        str(wav_path),
        task = task,
        language = language, # None -> auto
        **q_params,
    )

    segments: List[Segment] = []

    for segment in segments_iter:
        text = segment.text.strip()
        if not text: 
            continue

        avg_logprob = getattr(segment, "avg_logprob", None)
        no_speech_prob = getattr(segment, "no_speech_prob", None)

        segments.append(
            Segment(
                start = float(segment.start),
                end = float(segment.end),
                text = text,
                lang = info.language,
                avg_logprob = float(avg_logprob) if avg_logprob is not None else None,
                no_speech_prob = float(no_speech_prob) if no_speech_prob is not None else None,
            )
        )

    console.log(
        f"[blue] Got {len(segments)} segments"
        f"language = {info.language}, duration = {info.duration:.1f}s[/]")

    # SRT(optional)
    if output_srt is not None: 
        write_srt(
            [
                dict(
                    start = s.start, 
                    end = s.end,
                    text = s.text,
                )
                for s in segments
            ],
            output_srt,
        )
        console.log(f"[green] Written[/] SRT -> {output_srt}")

    # JSON(optional
    if output_json is not None:
        import json
        
        data = {
            "language": info.language,
            "duration": info.duration,
            "segments": [
                dict(
                    start = s.start,
                    end = s.end,
                    text = s.text,
                    lang = s.lang,
                    avg_logprob = s.avg_logprob, 
                    no_speech_prob = s.no_speech_prob,
                )
                for s in segments
            ],
        }
        output_json.parent.mkdir(parents = True, exist_ok = True)
        output_json.write_text(
            json.dumps(data, ensure_ascii = False, indent = 2),
            encoding = "utf-8",
        )
        console.log(f"[green] Written[/] JSON -> {output_json}")

    return {
        "language": info.language,
        "duration": info.duration,
        "segments_count": len(segments),
        "srt": str(output_srt),
        "json": str(output_json),
    }