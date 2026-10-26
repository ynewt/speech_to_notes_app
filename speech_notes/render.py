"""Turning structured notes into a Markdown document."""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from .chunking import segment_line
from .extract import Notes
from .transcript import Segment, format_ts


def _ts(item_ts: str) -> str:
    return f" _({item_ts})_" if item_ts else ""


def render_markdown(
    notes: Notes,
    title: str,
    source: str,
    model: str,
    segments: List[Segment],
    include_transcript: bool = False,
) -> str:
    duration: Optional[float] = segments[-1].end if segments else None
    meta = [f"Source: `{source}`"]
    if duration:
        meta.append(f"Duration: {format_ts(duration)}")
    meta += [f"Model: {model}", f"Generated: {date.today().isoformat()}"]

    out = [f"# {title}", "", "_" + " · ".join(meta) + "_", "", "## Summary", "", notes["summary"] or "_No summary._", ""]

    out += ["## Action items", ""]
    if notes["action_items"]:
        for a in notes["action_items"]:
            owner = f"**{a['owner']}**: " if a.get("owner") else ""
            due = f" (due: {a['due']})" if a.get("due") else ""
            out.append(f"- [ ] {owner}{a.get('task', '')}{due}{_ts(a.get('timestamp', ''))}")
    else:
        out.append("_None identified._")
    out.append("")

    out += ["## Decisions", ""]
    if notes["decisions"]:
        out += [f"- {d.get('decision', '')}{_ts(d.get('timestamp', ''))}" for d in notes["decisions"]]
    else:
        out.append("_None identified._")
    out.append("")

    out += ["## Open questions", ""]
    if notes["open_questions"]:
        out += [f"- {q}" for q in notes["open_questions"]]
    else:
        out.append("_None identified._")
    out.append("")

    if include_transcript:
        out += ["<details>", "<summary>Full transcript</summary>", ""]
        out += [segment_line(s) + "  " for s in segments]
        out += ["", "</details>", ""]

    return "\n".join(out)
