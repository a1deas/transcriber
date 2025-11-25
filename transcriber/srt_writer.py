# transcriber/transcriber/srt_writer.py
from pathlib import Path
from typing import Iterable

import logging

logger = logging.getLogger(__name__)

# float seconds -> SRT timecode | 00:01:23,456 |
def format_timestamp(timestamp: float) -> str: 
    if timestamp < 0:
        timestamp = 0.0
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    millis = int(round((timestamp - int(timestamp)) * 1000))

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def write_srt(segments: Iterable[dict], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    logger.info(f"[SRT Writer] Trying to write subtitles...")

    lines = []
    for idx, segment in enumerate(segments, start = 1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()

        if not text:
            continue

        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    
    logger.info(f"[SRT Writer] Done! Subtitles written to {output_path}")
    output_path.write_text("\n".join(lines), encoding="utf-8")