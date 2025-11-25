from pathlib import Path
import pytest

from transcriber.srt_writer import write_srt

def test(tmp_path: Path): 
    segments = [
        {"start": 0.0, "end": 1.5, "text": "Hello", "lang": "en"},
        {"start": 1.5, "end": 3.0, "text": "World", "lang": "en"},
    ]

    out = tmp_path / test.srt
    write_srt(segments, out)

    content = out.read_text(encoding = "utf-8")
    assert "Hello" in content
    assert "world" in content