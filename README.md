# Speech-to-Notes

A small, fully local tool that turns recordings (lab meetings, talks, voice memos) into structured Markdown notes with a **summary, action items, decisions, and open questions**.

It chains two foundation models:

1. **Whisper** (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)), a speech-recognition model, turns audio into timestamped text.
2. **A local LLM** (via [Ollama](https://ollama.com), `llama3.2:3b` by default) reads the transcript and extracts structured notes as JSON. The tool then renders the JSON as Markdown.

Nothing leaves your machine, and everything runs on a CPU.

## What you learn from this project

- **Chaining models into a pipeline.** The output of one model (text) becomes the input of another. Errors carry forward: if Whisper mishears "DE analysis" as "DNA", the LLM can't fix it.
- **Handling inputs longer than the context window.** An hour of speech is roughly 12,000 tokens, more than a small local model can take in at once. The tool uses a **map-reduce** strategy to handle this (see below).
- **Getting structured output from an LLM.** It passes a JSON schema to Ollama, which constrains the model so it always returns parseable JSON.
- **The practical limits of small models.** You can compare real token counts against estimates, see how long each stage takes, and see where a 3B model misses things.

## How it works

```
 audio file
     │
     ▼
┌─────────────────────┐   lab_meeting.transcript.json  (cached; reuse to skip this step)
│ 1. Whisper (STT)    │──▶ lab_meeting.transcript.txt   (human-readable)
└─────────────────────┘
     │  timestamped segments: [00:01:16] David, can you check the traces...
     ▼
┌─────────────────────┐
│ 2. Chunking         │  split into ~1500-token pieces along segment boundaries,
└─────────────────────┘  with ~150 tokens of overlap between neighbours
     │  chunk 1, chunk 2, ... chunk N
     ▼
┌─────────────────────┐
│ 3. Map (LLM)        │  extract {summary, decisions, action_items, open_questions}
└─────────────────────┘  from each chunk independently, as schema-constrained JSON
     │  N partial notes
     ▼
┌─────────────────────┐
│ 4. Reduce (LLM)     │  merge partials, remove duplicates caused by overlap, and
└─────────────────────┘  write an overall summary. If the partials don't fit in one
     │                   prompt, merge them in groups and repeat (hierarchical).
     ▼
 lab_meeting.notes.md
```

### Why chunk at all?

LLMs have a fixed **context window** (`num_ctx`): the most tokens they can attend to at once, counting prompt and answer together. With Ollama, if your prompt is longer than `num_ctx`, **the beginning of the prompt is silently dropped**. You get no error, just notes that leave out the first half of your meeting. The tool budgets for this:

```
chunk tokens (1500) + system prompt (~350) + room for the answer (~1200)  ≤  num_ctx (4096)
```

It warns you if your settings break this budget, and it warns again if Ollama reports that a prompt filled the whole context.

Token counts are estimated as `characters / 4`. After each call the tool logs Ollama's real count next to the estimate (`963 prompt tokens (est. 800)`), so you can see how rough the estimate is.

### Why overlap?

If someone says "David, can you send the request…" right at a chunk boundary, neither chunk may contain the whole sentence. Repeating a little text across boundaries prevents that. The trade-off is that items in the overlap get extracted twice, which is why the reduce step deduplicates.

### Two ways to merge (`--merge`)

| Mode | How it works | Pros | Cons |
|---|---|---|---|
| `llm` (default) | Asks the LLM to merge partial notes and rewrite the summary | Merges near-duplicates, makes one coherent summary | Small models sometimes drop or garble items |
| `simple` | Concatenates the lists and drops exact duplicates | Fast, never loses items | Keeps near-duplicates; the summary is just partial summaries joined together |

## Setup (macOS)

### 1. Python 3.9 or newer

faster-whisper needs Python ≥ 3.9. Check yours with `python3 --version`. If it's older, install a newer one:

```bash
brew install python@3.12
```

### 2. Create a virtual environment and install dependencies

```bash
git clone https://github.com/ynewt/speech_to_notes_app.git
cd speech_to_notes_app
```

Then, from the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs `faster-whisper`, which brings in CTranslate2 (the inference engine) and PyAV (which decodes audio, so a separate ffmpeg install isn't needed). The first transcription downloads the Whisper model weights from Hugging Face, about 150 MB for `base`. You can ignore the warning about unauthenticated Hugging Face requests.

### 3. Install Ollama and pull a model

```bash
brew install ollama        # or download the app from https://ollama.com
ollama serve               # leave running in another terminal (the desktop app does this for you)
ollama pull llama3.2:3b
```

Check that it works: `ollama run llama3.2:3b "Say hi"`.

### 4. Run the tests (optional)

```bash
python -m pytest -q
```

The tests cover chunking, transcript I/O, merging, and rendering. They don't need Whisper or Ollama.

## Usage

Make sure the virtual environment is active (`source .venv/bin/activate`) and Ollama is running.

### Audio to notes

```bash
python -m speech_notes recording.m4a
```

This writes three files next to the input:

- `recording.transcript.json`: the transcript with timestamps, which you can feed back in
- `recording.transcript.txt`: the same transcript, easy to read
- `recording.notes.md`: the notes

Supported audio formats include `.m4a`, `.mp3`, `.wav`, `.aiff`, `.flac`, `.ogg`, `.webm`, `.mp4`, and `.mov`.

### Skip transcription (fast iteration on the LLM step)

Transcription is the slow, deterministic part. Run it once, then experiment with prompts, models, and chunk sizes on the saved transcript:

```bash
python -m speech_notes recording.transcript.json --llm qwen2.5:1.5b --merge simple
```

Plain-text transcripts work too, but without timestamps:

```bash
python -m speech_notes examples/lab_meeting.txt
```

### Try it without a recording

`examples/lab_meeting.txt` is a fictional lab meeting script. Turn it into audio with the macOS `say` command to test the whole pipeline:

```bash
say -r 200 -o lab_meeting.aiff -f examples/lab_meeting.txt
python -m speech_notes lab_meeting.aiff --chunk-tokens 600 --language en
```

`--chunk-tokens 600` forces this 3-minute recording into 3 chunks, so you can watch the map and reduce steps run.

### Other examples

```bash
# Better transcription quality (slower), English only, custom output path and title
python -m speech_notes talk.mp3 --whisper-model small --language en -o notes/talk.md --title "Seminar: Dr. Lee"

# Just transcribe
python -m speech_notes memo.m4a --transcribe-only

# Bigger context window = fewer chunks (uses more RAM and is slower per call)
python -m speech_notes meeting.transcript.json --num-ctx 8192 --chunk-tokens 5000

# Put the full transcript at the bottom of the notes (collapsed)
python -m speech_notes meeting.m4a --include-transcript
```

### All options

| Option | Default | Description |
|---|---|---|
| `input` | (required) | Audio file, or a `.transcript.json` / `.txt` transcript |
| `-o, --output` | `<input>.notes.md` | Output Markdown path |
| `--title` | from file name | Title at the top of the notes |
| `--whisper-model` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`, … |
| `--language` | auto-detect | Language code such as `en`; setting it skips detection and avoids mistakes |
| `--transcribe-only` | off | Stop after writing the transcript |
| `--llm` | `llama3.2:3b` | Any model you've pulled with Ollama |
| `--ollama-host` | `http://localhost:11434` | Ollama server URL |
| `--num-ctx` | `4096` | LLM context window in tokens |
| `--chunk-tokens` | `1500` | Maximum transcript tokens per chunk |
| `--overlap-tokens` | `150` | Tokens repeated between neighbouring chunks |
| `--merge` | `llm` | `llm` or `simple` (see above) |
| `--include-transcript` | off | Append the full transcript to the notes |

## Example output

Produced from `examples/lab_meeting.txt` (via `say` → Whisper `base` → `llama3.2:3b`, 3 chunks):

```markdown
## Action items

- [ ] **David**: Check the bioanalyzer traces and send the re-sequencing request to the core by Friday (due: Friday) _(00:01:16)_
- [ ] **Priya**: Make draft versions of RNA figures by November 7th (due: November 7th) _(00:01:57)_
- [ ] **David**: Make an inventory spreadsheet beforehand (due: Tuesday evening) _(00:02:45)_
...

## Decisions

- Re-sequence the two libraries instead of re-extracting them _(00:01:16)_
```

It's good but not perfect, and the gaps are worth studying. The model **missed the decision** to move samples to another lab's freezer, and the speaker's recap at the end produced **near-duplicate** action items. Things to try: a larger model, `--merge simple`, bigger chunks, or editing the prompts in `speech_notes/extract.py`.

## Performance

Measured on a 2019 MacBook Pro (Intel i7-8557U, 4 cores, CPU only) with a 3-minute recording:

| Stage | Time |
|---|---|
| Whisper `base` transcription (int8) | ~45 s, about 4× faster than real time |
| `llama3.2:3b`, per ~600-token chunk | ~40–60 s |
| `llama3.2:3b`, merge step | ~80 s |

Rough guidance:

- **Whisper models:** `tiny`/`base` are fast and fine for clear audio. `small` is noticeably more accurate on real-world audio and about 3× slower. `medium`/`large-v3` are best but slow on a CPU. On Apple Silicon everything runs much faster.
- **LLMs:** `llama3.2:3b` is a good default. `qwen2.5:1.5b` and `llama3.2:1b` are faster but miss more. If you have 16 GB+ RAM, try a 7–8B model (`ollama pull qwen2.5:7b` or `llama3.1:8b`) for clearly better extraction.
- A one-hour meeting with the defaults means about 8 chunks plus a merge, so expect around 10–15 minutes on a CPU like the one above.

## Project layout

```
speech_notes/
  transcript.py   stage 1: faster-whisper transcription; save/load transcripts
  chunking.py     token estimation and overlapping chunking along segment boundaries
  llm.py          minimal Ollama HTTP client (stdlib only); truncation warning
  extract.py      stage 2: prompts, JSON schema, map step, simple and hierarchical LLM reduce
  render.py       structured notes → Markdown
  cli.py          command-line interface tying it all together
examples/
  lab_meeting.txt sample transcript for testing
tests/
  test_pipeline.py
```

## Ideas for extending it

- **Speaker diarization** ("who said what") with [pyannote.audio](https://github.com/pyannote/pyannote-audio), merged with Whisper segments by timestamp, so the LLM gets `Maria: ...` labels even from real audio.
- **Compare merge strategies.** Write a small evaluation: hand-label the action items in `examples/lab_meeting.txt` and measure recall for each model, chunk size, and merge mode.
- **Refine instead of map-reduce.** Process chunks in order and pass along a running set of notes ("here are the notes so far; update them with this excerpt"). This keeps context between chunks but can't run in parallel.
- **Parallelize the map step.** Set `OLLAMA_NUM_PARALLEL` and send chunk requests concurrently.
- **Watch a folder** of voice memos and process new files automatically.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not reach Ollama at http://localhost:11434` | Start it: `ollama serve`, or open the Ollama app |
| `Model 'xyz' is not pulled` | `ollama pull xyz` |
| `WARNING: prompt used N tokens, filling the whole … context` | Input was truncated. Lower `--chunk-tokens` or raise `--num-ctx` |
| `TypeError: open() got an unexpected keyword argument 'metadata_errors'` | PyAV is too new for faster-whisper: `pip install "av<19"` (already pinned in `requirements.txt`) |
| `pip install` fails building packages | Your Python is probably too old. Use Python ≥ 3.9 |
| Transcript has repeated phrases or text that was never said during silence | Classic Whisper hallucination. Try a larger `--whisper-model` and set `--language` |
| Notes are wrong because the transcript is wrong | Check `*.transcript.txt` first; errors carry forward from stage 1 |
