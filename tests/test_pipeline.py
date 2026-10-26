"""Tests for the parts of the pipeline that don't need Whisper or an LLM."""

import json

from speech_notes.chunking import chunk_segments, estimate_tokens, segment_line
from speech_notes.extract import empty_notes, simple_merge
from speech_notes.render import render_markdown
from speech_notes.transcript import Segment, format_ts, load_transcript, save_transcript


def make_segments(n, words_per_segment=20):
    return [Segment(i * 5.0, i * 5.0 + 5, " ".join(["word"] * words_per_segment) + f" {i}.") for i in range(n)]


def test_format_ts():
    assert format_ts(3725.9) == "01:02:05"
    assert format_ts(None) == ""


def test_short_transcript_is_one_chunk():
    chunks = chunk_segments(make_segments(3), max_tokens=1000)
    assert len(chunks) == 1


def test_chunks_respect_budget_and_cover_everything():
    segments = make_segments(200)
    chunks = chunk_segments(segments, max_tokens=300, overlap_tokens=50)
    assert len(chunks) > 1
    for c in chunks:
        assert c.tokens <= 300 + len(chunks[0].segments)  # newline joins add a little slack
    covered = {id(s) for c in chunks for s in c.segments}
    assert covered == {id(s) for s in segments}


def test_chunks_overlap():
    chunks = chunk_segments(make_segments(100), max_tokens=300, overlap_tokens=60)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.segments[0] in prev.segments


def test_long_segment_is_split():
    big = Segment(None, None, " ".join(["lorem"] * 2000))
    chunks = chunk_segments([big], max_tokens=200, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(estimate_tokens(segment_line(s)) <= 200 for c in chunks for s in c.segments)


def test_transcript_roundtrip(tmp_path):
    segments = make_segments(3)
    path = tmp_path / "x.transcript.json"
    txt = save_transcript(segments, path)
    assert load_transcript(path) == segments
    assert txt.read_text().startswith("[00:00:00] word")


def test_plain_text_is_split_into_sentences(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("Hello there. How are you?\n\nSecond paragraph!")
    assert [s.text for s in load_transcript(path)] == ["Hello there.", "How are you?", "Second paragraph!"]


def test_simple_merge_dedupes():
    a, b = empty_notes(), empty_notes()
    a["action_items"] = [{"task": "Send the request.", "owner": "David", "due": "", "timestamp": ""}]
    b["action_items"] = [{"task": "send the request", "owner": "David", "due": "", "timestamp": ""},
                         {"task": "Make figures", "owner": "Priya", "due": "", "timestamp": ""}]
    a["summary"], b["summary"] = "Part one.", "Part two."
    merged = simple_merge([a, b])
    assert [x["task"] for x in merged["action_items"]] == ["Send the request.", "Make figures"]
    assert merged["summary"] == "Part one. Part two."


def test_render_markdown():
    notes = empty_notes()
    notes["summary"] = "We met."
    notes["action_items"] = [{"task": "Do it", "owner": "Ana", "due": "Friday", "timestamp": "00:01:00"}]
    md = render_markdown(notes, "Notes", "a.m4a", "m", make_segments(2))
    assert "- [ ] **Ana**: Do it (due: Friday) _(00:01:00)_" in md
    assert "## Decisions" in md
    json.dumps(notes)  # notes stay JSON-serialisable
