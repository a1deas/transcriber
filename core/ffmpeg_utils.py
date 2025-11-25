# transcriber/core/ffmpeg_utils.py
from pathlib import Path
import ffmpeg
from typing import Optional

# Extrancts audio and converts it to WAV 16Hz mono.
# If its already WAV, normalize it via FFmpeg anyway.
def extract_wav(
        input_path: Path, 
        output_path: Path,
        sample_rate: int = 16000, 
        ac: int = 1,
) -> Path: 
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents = True, exist_ok = True)

    (
        ffmpeg
        .input(str(input_path))
        .output(
            str(output_path),
            ac = ac, 
            ar = sample_rate, 
            formar = "wav",
        )
        .overwrite_output()
        .run(quiet = True)
    )

    return output_path