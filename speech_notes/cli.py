"""Command-line entry point: audio -> transcript -> chunks -> notes -> Markdown."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .chunking import chunk_segments
from .extract import OUTPUT_RESERVE_TOKENS, SYSTEM_PROMPT_TOKENS, extract_notes
from .llm import LLMError, OllamaClient
from .render import render_markdown
from .transcript import is_audio, load_transcript, log, save_transcript, transcribe


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m speech_notes",
        description="Transcribe audio with Whisper and extract action items and decisions with a local LLM.",
    )
    p.add_argument("input", type=Path, help="audio file, or a transcript (.transcript.json or .txt) to skip transcription")
    p.add_argument("-o", "--output", type=Path, help="output Markdown file (default: <input>.notes.md)")
    p.add_argument("--title", help="title for the notes (default: derived from file name)")

    w = p.add_argument_group("transcription (faster-whisper)")
    w.add_argument("--whisper-model", default="base", help="tiny, base, small, medium, large-v3, ... (default: base)")
    w.add_argument("--language", help="language code such as 'en' (default: auto-detect)")
    w.add_argument("--transcribe-only", action="store_true", help="stop after writing the transcript")

    l = p.add_argument_group("note extraction (Ollama)")
    l.add_argument("--llm", default="llama3.2:3b", help="Ollama model name (default: llama3.2:3b)")
    l.add_argument("--ollama-host", default="http://localhost:11434")
    l.add_argument("--num-ctx", type=int, default=4096, help="LLM context window in tokens (default: 4096)")
    l.add_argument("--chunk-tokens", type=int, default=1500, help="max transcript tokens per chunk (default: 1500)")
    l.add_argument("--overlap-tokens", type=int, default=150, help="tokens repeated between chunks (default: 150)")
    l.add_argument("--merge", choices=["llm", "simple"], default="llm",
                   help="how to combine per-chunk notes: 'llm' dedupes and rewrites, 'simple' just concatenates")
    l.add_argument("--include-transcript", action="store_true", help="append the full transcript to the notes")
    return p.parse_args(argv)


def base_name(path: Path) -> str:
    name = path.name
    for suffix in (".transcript.json", ".transcript.txt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        log(f"Input not found: {args.input}")
        return 1

    stem = base_name(args.input)
    output = args.output or args.input.with_name(f"{stem}.notes.md")
    t0 = time.time()

    # Stage 1: speech -> text
    if is_audio(args.input):
        segments = transcribe(args.input, args.whisper_model, args.language)
        transcript_path = output.with_name(f"{stem}.transcript.json")
        txt_path = save_transcript(segments, transcript_path)
        log(f"Transcript saved to {transcript_path} and {txt_path} ({time.time() - t0:.0f}s)")
        if args.transcribe_only:
            return 0
    else:
        segments = load_transcript(args.input)
        log(f"Loaded transcript with {len(segments)} segments from {args.input}")

    if not segments:
        log("Transcript is empty; nothing to do.")
        return 1

    # Stage 2: text -> structured notes
    needed = args.chunk_tokens + SYSTEM_PROMPT_TOKENS + OUTPUT_RESERVE_TOKENS
    if needed > args.num_ctx:
        log(f"WARNING: --chunk-tokens {args.chunk_tokens} + prompt + answer (~{needed}) exceeds --num-ctx {args.num_ctx}.")

    chunks = chunk_segments(segments, args.chunk_tokens, args.overlap_tokens)
    total_tokens = sum(c.tokens for c in chunks)
    log(f"Split transcript into {len(chunks)} chunk(s), ~{total_tokens} tokens total")

    client = OllamaClient(args.llm, args.ollama_host, args.num_ctx)
    t1 = time.time()
    try:
        client.check()
        notes = extract_notes(client, chunks, args.merge)
    except LLMError as e:
        log(f"Error: {e}")
        return 1
    log(f"Extraction took {time.time() - t1:.0f}s")

    title = args.title or f"Notes: {stem.replace('_', ' ').replace('-', ' ')}"
    md = render_markdown(notes, title, args.input.name, args.llm, segments, args.include_transcript)
    output.write_text(md)
    log(f"Notes written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
