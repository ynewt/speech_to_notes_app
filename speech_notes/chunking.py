"""Splitting long transcripts into pieces that fit the LLM's context window.

A one-hour meeting is ~9,000 words, roughly 12,000 tokens -- more than a small
local model can attend to at once (and Ollama silently truncates prompts that
exceed num_ctx). So we cut the transcript into chunks along segment boundaries,
with a little overlap so an action item spoken across a boundary is not lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .transcript import Segment, format_ts


def estimate_tokens(text: str) -> int:
    """Rough token count. ~4 characters per token holds for English with most
    tokenizers; good enough for budgeting. (Ollama reports the real count after
    each call, which the pipeline logs so you can compare.)"""
    return len(text) // 4 + 1


def segment_line(seg: Segment) -> str:
    """How a segment appears in the prompt: timestamps let the LLM cite where things were said."""
    if seg.start is None:
        return seg.text
    return f"[{format_ts(seg.start)}] {seg.text}"


@dataclass
class Chunk:
    index: int
    segments: List[Segment]

    @property
    def text(self) -> str:
        return "\n".join(segment_line(s) for s in self.segments)

    @property
    def start(self) -> Optional[float]:
        return self.segments[0].start

    @property
    def end(self) -> Optional[float]:
        return self.segments[-1].end

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


def _split_long_segment(seg: Segment, max_tokens: int) -> List[Segment]:
    """Break a single over-long segment (e.g. a huge paragraph) into word groups."""
    words = seg.text.split()
    pieces, current = [], []
    for word in words:
        if current and estimate_tokens(" ".join(current + [word])) > max_tokens:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return [Segment(seg.start, seg.end, p) for p in pieces]


def chunk_segments(segments: List[Segment], max_tokens: int = 1500, overlap_tokens: int = 150) -> List[Chunk]:
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    units: List[Segment] = []
    for seg in segments:
        if estimate_tokens(segment_line(seg)) > max_tokens:
            units.extend(_split_long_segment(seg, max_tokens - 10))
        else:
            units.append(seg)

    groups: List[List[Segment]] = []
    current: List[Segment] = []
    current_tokens = 0
    for seg in units:
        t = estimate_tokens(segment_line(seg))
        if current and current_tokens + t > max_tokens:
            groups.append(current)
            # Seed the next chunk with the tail of this one (the overlap).
            tail: List[Segment] = []
            tail_tokens = 0
            for s in reversed(current):
                st = estimate_tokens(segment_line(s))
                if tail_tokens + st > overlap_tokens:
                    break
                tail.insert(0, s)
                tail_tokens += st
            current, current_tokens = tail, tail_tokens
        current.append(seg)
        current_tokens += t
    if current:
        groups.append(current)

    return [Chunk(i, g) for i, g in enumerate(groups)]
