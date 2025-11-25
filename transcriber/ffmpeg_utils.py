# transcriber/transcriber/ffmpeg_utils.py
from pathlib import Path
from typing import Optional
import ffmpeg
import logging

logger = logging.getLogger(__name__)

# Extrancts audio and converts it to WAV 16kHz mono.
# If its already WAV, normalize it via FFmpeg anyway.
def extract_wav(
        input_path: Path, 
        output_path: Path,
        sample_rate: int = 16000, 
        ac: int = 1,
) -> Path: 
    
    logger.info(f"[Audio Extraction] Trying to extract WAV from {input_path}")

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
            format = "wav",
        )
        .overwrite_output()
        .run(quiet = True)
    )

    logger.info(f"[Audio Extraction] Done! Audio extracted to {output_path} | sample rate = {sample_rate} | ac = {ac}")

    return output_path