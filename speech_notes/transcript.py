"""Stage 1 of the pipeline: speech -> timestamped text with faster-whisper.

Also handles saving/loading transcripts so the (slow) transcription step only
has to run once while you experiment with the LLM step.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aiff", ".aif", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mov", ".mkv"}


@dataclass
class Segment:
    start: Optional[float]  # seconds; None for plain-text transcripts
    end: Optional[float]
    text: str


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def format_ts(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def transcribe(
    audio_path: Path,
    model_size: str = "base",
    language: Optional[str] = None,
    compute_type: str = "int8",
    beam_size: int = 5,
) -> List[Segment]:
    # Imported lazily so the LLM half of the tool works without faster-whisper installed.
    from faster_whisper import WhisperModel

    log(f"Loading Whisper model '{model_size}' (first run downloads it)...")
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    # transcribe() returns a lazy generator: the actual decoding happens while we
    # iterate over it, which is why progress is reported inside the loop.
    # vad_filter skips silence, which speeds things up and reduces hallucinations.
    segments_iter, info = model.transcribe(
        str(audio_path), language=language, beam_size=beam_size, vad_filter=True
    )
    log(
        f"Detected language: {info.language} (p={info.language_probability:.2f}), "
        f"duration {format_ts(info.duration)}"
    )

    segments = []
    for seg in segments_iter:
        segments.append(Segment(seg.start, seg.end, seg.text.strip()))
        pct = 100 * seg.end / info.duration if info.duration else 0.0
        print(f"\r  transcribing... {min(pct, 100):5.1f}%", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return segments


def save_transcript(segments: List[Segment], json_path: Path) -> Path:
    """Write <name>.transcript.json (reloadable) and <name>.transcript.txt (readable)."""
    json_path.write_text(json.dumps([asdict(s) for s in segments], indent=2, ensure_ascii=False))
    txt_path = json_path.with_suffix(".txt")
    lines = [f"[{format_ts(s.start)}] {s.text}" if s.start is not None else s.text for s in segments]
    txt_path.write_text("\n".join(lines) + "\n")
    return txt_path


def load_transcript(path: Path) -> List[Segment]:
    """Load a transcript saved by save_transcript(), or any plain-text file."""
    if path.suffix.lower() == ".json":
        return [Segment(**d) for d in json.loads(path.read_text())]

    # Plain text: no timestamps, so split into sentences to give the chunker
    # small units to work with.
    text = path.read_text()
    segments = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if sentence:
                segments.append(Segment(None, None, sentence))
    return segments
