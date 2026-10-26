"""Stage 2 of the pipeline: transcript chunks -> structured notes with a local LLM.

Strategy: map-reduce.
  map:    extract notes from each chunk independently (each fits in context).
  reduce: merge the partial notes into one set, removing duplicates caused by
          chunk overlap. If the partial notes are themselves too long to fit in
          one prompt, merge them in groups and repeat (hierarchical reduce).
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

from .chunking import Chunk, estimate_tokens
from .llm import LLMError, OllamaClient
from .transcript import format_ts, log

Notes = Dict[str, object]

# JSON schema passed to Ollama so the model's output is guaranteed to parse.
NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"decision": {"type": "string"}, "timestamp": {"type": "string"}},
                "required": ["decision", "timestamp"],
            },
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": "string"},
                    "due": {"type": "string"},
                    "timestamp": {"type": "string"},
                },
                "required": ["task", "owner", "due", "timestamp"],
            },
        },
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "decisions", "action_items", "open_questions"],
}

MAP_SYSTEM = """You are a careful note-taker. You will receive one excerpt from a transcript of a meeting, talk, or voice memo. Lines may start with [HH:MM:SS] timestamps. Transcripts come from speech recognition and may contain small errors.

Extract ONLY what is actually said in the excerpt:
- summary: 2-4 sentences describing what this excerpt covers.
- decisions: conclusions or choices the speakers agreed on.
- action_items: concrete tasks someone committed to or was assigned. owner = the person's name if stated, otherwise "". due = the deadline if stated, otherwise "".
- open_questions: questions or issues raised but not resolved.
For timestamp, copy the [HH:MM:SS] of the line where the item appears, without brackets, or "" if there are no timestamps.

Do not invent items. Empty lists are fine. Respond with JSON only."""

REDUCE_SYSTEM = """You merge partial notes that were extracted from consecutive excerpts of ONE transcript, given in order. Neighbouring excerpts overlap slightly, so the same item can appear twice with different wording.

- summary: write 3-6 sentences summarising the whole recording, using the partial summaries.
- decisions, action_items, open_questions: include every item from the input, merging only true duplicates (keep the earliest timestamp and the most complete wording). Drop an open question if a later decision resolves it.

Do not invent items. Respond with JSON only."""

# Token budgeting: the prompt must leave room for the system prompt and the answer.
SYSTEM_PROMPT_TOKENS = 350
OUTPUT_RESERVE_TOKENS = 1200


def empty_notes() -> Notes:
    return {"summary": "", "decisions": [], "action_items": [], "open_questions": []}


def _parse(content: str) -> Notes:
    notes = empty_notes()
    data = json.loads(content)
    for key, default in notes.items():
        value = data.get(key, default)
        notes[key] = value if isinstance(value, type(default)) else default
    return notes


def _call(client: OllamaClient, system: str, user: str, label: str) -> Notes:
    for attempt in (1, 2):
        resp = client.chat(system, user, schema=NOTES_SCHEMA)
        log(
            f"  {label}: {resp.prompt_tokens} prompt tokens (est. {estimate_tokens(system + user)}), "
            f"{resp.output_tokens} output tokens, {resp.seconds:.1f}s"
        )
        try:
            return _parse(resp.content)
        except (json.JSONDecodeError, AttributeError):
            log(f"  {label}: could not parse model output (attempt {attempt})")
    raise LLMError(f"{label}: model did not return valid JSON")


# --- map -------------------------------------------------------------------

def extract_chunk(client: OllamaClient, chunk: Chunk, total: int) -> Notes:
    span = ""
    if chunk.start is not None:
        span = f", covering {format_ts(chunk.start)}-{format_ts(chunk.end)}"
    user = f"Transcript excerpt {chunk.index + 1} of {total}{span}:\n\n{chunk.text}"
    return _call(client, MAP_SYSTEM, user, f"chunk {chunk.index + 1}/{total}")


# --- reduce ----------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def simple_merge(partials: List[Notes]) -> Notes:
    """Deterministic merge: concatenate and drop exact (normalised) duplicates.
    Never loses items, but keeps near-duplicates and just joins the summaries."""
    merged = empty_notes()
    merged["summary"] = " ".join(p["summary"] for p in partials if p["summary"])
    seen = set()
    for p in partials:
        for key, text_field in (("decisions", "decision"), ("action_items", "task"), ("open_questions", None)):
            for item in p[key]:
                text = item if text_field is None else str(item.get(text_field, ""))
                fingerprint = (key, _norm(text))
                if text and fingerprint not in seen:
                    seen.add(fingerprint)
                    merged[key].append(item)
    return merged


def _group_to_fit(partials: List[Notes], budget: int) -> List[List[Notes]]:
    groups: List[List[Notes]] = [[]]
    used = 0
    for p in partials:
        t = estimate_tokens(json.dumps(p))
        if groups[-1] and used + t > budget:
            groups.append([])
            used = 0
        groups[-1].append(p)
        used += t
    return groups


def llm_merge(client: OllamaClient, partials: List[Notes]) -> Notes:
    budget = client.num_ctx - SYSTEM_PROMPT_TOKENS - OUTPUT_RESERVE_TOKENS
    level = 1
    while len(partials) > 1:
        groups = _group_to_fit(partials, budget)
        if len(groups) == len(partials):
            # Every partial is too big to pair with another; the LLM can't make progress.
            log("  partial notes too large to merge with the LLM; falling back to simple merge")
            return simple_merge(partials)
        merged = []
        for i, group in enumerate(groups):
            if len(group) == 1:
                merged.append(group[0])
                continue
            user = "\n\n".join(
                f"Partial notes {j + 1} of {len(group)}:\n{json.dumps(p, ensure_ascii=False)}"
                for j, p in enumerate(group)
            )
            merged.append(_call(client, REDUCE_SYSTEM, user, f"merge L{level} group {i + 1}/{len(groups)}"))
        partials = merged
        level += 1
    return partials[0]


def extract_notes(client: OllamaClient, chunks: List[Chunk], merge: str = "llm") -> Notes:
    log(f"Extracting notes from {len(chunks)} chunk(s) with {client.model}...")
    partials = [extract_chunk(client, c, len(chunks)) for c in chunks]

    if len(partials) == 1:
        return partials[0]
    log(f"Merging {len(partials)} partial notes ({merge})...")
    if merge == "simple":
        return simple_merge(partials)
    return llm_merge(client, partials)
