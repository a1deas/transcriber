# transcriber/transcriber/cli.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .api import transcribe_file

console = Console()

app = typer.Typer(
    help = "Simple CLI for audio/video transcription via faster-whisper")

# Video/Audio transcription to SRT/JSON
@app.command()
def run(
    input_path: Path = typer.Argument(
        ...,
        help = "Path to input video/audio file",
    ),
    output_srt: Path = typer.Option(
        "output.srt",
        "--srt",
        help = "Path to SRT-file",
    ),
    output_json: Path = typer.Option(
        "output.json",
        "--json",
        help = "Path to JSON-file",
    ),
    model: str = typer.Option(
        "small",
        "--model",
        "-m",
        help = "Faster-whisper model's name (small, medium, large, ... or path)",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        "-d",
        help = "cuda / cpu",
    ),
    compute_type: str = typer.Option(
        "float16",
        "--compute-type",
        help = "float16 / float32 / int8 / int8_float16 / int8_bfloat16",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help = "Language code ('en', 'ru', 'fr', etc.). If not set — auto.",
    ),
    task: str = typer.Option(
        "transcribe",
        "--task",
        "-t",
        help = "transcribe / translate",
    ),
    quality: str = typer.Option(
        "balanced",
        "--quality",
        "-q",
        help = "Quality profile: fast / balanced / quality",
    ),
):
    if not input_path.exists():
        raise typer.BadParameter(f"Input file does not exist: {input_path}")
    
    if task not in ("transcribe", "translate"):
        raise typer.BadParameter(f"Task must be either 'transcribe' or 'translate'")
    
    if quality not in ("fast", "balanced", "quality"):
        raise typer.BadParameter("Quality must be 'fast', 'balanced' or 'quality'")

    console.log(f"[bold cyan] Transcriber[/] running on [yellow]{input_path}[/]")

    transcription = transcribe_file(
        input_path = input_path,
        model_name = model,
        device = device,
        compute_type = compute_type,
        language = language,
        task = task,          # type: ignore[arg-type]
        quality = quality,  # type: ignore[arg-type]
        output_srt = output_srt,
        output_json = output_json,
    )

    console.log(
        f"[green]Done[/]: Language: {transcription['language']} | "
        f"Duration: {transcription['duration']:.1f}s | "
        f"Segments: {transcription['segments_count']} | "
        f".srt: '{transcription['srt']}', .json: '{transcription['json']}'"
    )
