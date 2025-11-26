# transcriber/transcriber/srt_writer.py
from pathlib import Path
from typing import Iterable, Mapping, Any
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

def write_srt(
    segments: Iterable[dict], 
    output_path: Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    logger.info(f"[srt] Trying to write subtitles...")

    lines = []
    idx = 1

    for segment in segments:
        start = format_timestamp(float(segment["start"]))
        end = format_timestamp(float(segment["end"]))
        text = str(segment["text"]).strip()

        if not text:
            continue

        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
        idx += 1
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[SRT Writer] Done! Wrote {idx - 1} entries. "
                f"Subtitles written to {output_path}")