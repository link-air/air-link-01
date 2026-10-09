# air-link-01

A **relationship-memory system** for chat: it remembers not *only* the user, but *this interaction*.

> **How it works** — plain language first, the terms come in the next section:
>
> 1. **You say something.** The system does the *looking back* on her behalf: which past moments does
>    this touch, and should memory be consulted at all? What it finds is ranked by "how many cues hit
>    the same record", and only the top two or three are put in front of her.
> 2. **She speaks.** In the whole pipeline, only *how to say it* belongs to her. Things with a definite
>    answer (similarity, ordering, thresholds) are done by code — code is more reliable than a model —
>    while semantic judgement ("are these the same thing?") is hers.
> 3. **Only then does it write.** A stretch of conversation becomes one *scene card*; cards on the same
>    topic aggregate into a narrative; and only a pattern that keeps being corroborated (≥3 times, in
>    the same kind of situation) is allowed to become "how she sees you". New evidence revises it; say
>    "that's wrong" and it is really deleted.
> 4. **Unfinished matters are tracked separately**: when due she gets one reminder (one at a time), and
>    once raised it is not raised again — until the matter is resolved.
> 5. She **never speaks first** — her words only happen after yours.
> 6. Every step above **leaves a trace**: why this was recalled, and why that was not, are both
>    readable in the workbench.
>
> The Chinese [`README.md`](README.md) is canonical · module interfaces: [`docs/modules.md`](docs/modules.md)

**Zero dependencies**: Python 3.11+, standard library only — even the LLM and embedding calls are
plain `urllib`. **The split is strict**: anything with a determinate answer — similarity, ranking,
thresholds, whether a scene should be cut — is code (code beats a model at these); only the
**semantic judgements that need a brain** ("are these the same matter?") go to the model.
The UI is bilingual (Chinese / English, switched in Settings — her replies follow; memory *content*
is never translated, it is data).

## Core mechanism

**Four memory levels**: S0 raw transcripts (one file per day) → S1 scene cards → S2 topical summaries →
S3 portraits. The more abstract, the less reversible — hence the layering. `profiles` (her inferences)
and `user_facts` (what you stated) are stored separately: facts are never corroborated or aged out.

**The order of a reply is fixed**: recall (read) → assemble context → generate → write. Recall must
happen before the write, or the current message leaks into history — double injection, plus the cues
would see "the future". Broken down it is nine steps, and **she only appears at step 4 ("speak")**;
the other six are done by the system on her behalf (see
[`设计与实现对照-结构.md`](设计与实现对照-结构.md) §1, Chinese).

### Recall (read): cues first, actions second

Seven cues (**one lightweight LLM call per turn** judges C2/C3/C5/C7; C1/C4 use embeddings; C6 is a
lookup):

| Cue | Judges | Fires when | Why it exists |
|---|---|---|---|
| **C1 semantic** | top similarity against the whole store | > 0.40 (drops to 0.03 when embeddings are unavailable and it degrades to character overlap — **the threshold degrades with it**, otherwise degrading would mean amnesia) | **The main path**: which past moment does this resemble. Without it, recall could only be keyword-driven |
| **C2 tense** | where the sentence points | not `now` (past / future / hypothetical) | It decides the direction: "that thing last time" → look backwards (time travel); "interview tomorrow" → think forwards (topical search) |
| **C3 emotion** | valence + arousal | `arousal=1`; an uncertain valence means **do nothing** | When feelings run high, what should surface is a **moment of the same kind** (not a random one); it also says "this turn may carry a few more" (budget +3) |
| **C4 self-relevance** | similarity to "about this person" memories | > 0.70, **capped** — self-reference must not take over | This sentence is about *him* — worth looking back for, so it earns a vote; the cap keeps it from outshouting other cues |
| **C5 relation** | is this about the two of us | yes | It is about the two of you (agreements, how the relationship shifted) — the relationship is memory too, so it earns a vote |
| **C6 freshness** | 1 − top similarity | < 0.30 = "you have talked about this" | Only used to **say one thing**: "you have talked about this" — a signpost, not content |
| **C7 unresolved** | is this an open matter | yes (an attribute of *this sentence*, not stored) | It is about an **unfinished matter** — unfinished things deserve to be remembered (that is how people work too), so it earns a vote |

(C2–C7 are worth **1 vote each**: one alone decides nothing, but two or more together are the
"multi-dimensional" signal that pushes a record up the ranking and unlocks the raw text in R5.)

Two **bypasses** sit next to the cues and skip semantic search entirely: **entity** (a name that
carries a personal relation in the store) and **literal** (2–4 character *rare* fragments — appearing
in at most 3 scenes; no word list, "what matters" is counted, so 「我们」/「的时候」 are shut out by
frequency alone, max 3 words per message).

Cues couple into **actions** (rule-based, each with its own threshold; mathematically equivalent to a
weighted sum, and v1 never fits the weights):

| Action | When | What it does |
|---|---|---|
| **R0 perceive** | every input | pronouns / time words / emotion words / entity names / first person → look. **The default leans towards looking**: a missed recall (amnesia) is unrecoverable, an extra one is just noise |
| **R1 topical search** | C1 fires, or C2 = future / hypothetical | top 2N by similarity, then **one step of same-topic spread** (spread = "associated recall", it earns no votes) |
| **R2 time travel** | C2 = past | the most recent 60, but they **must pass the weak-relevance gate** — "past" carries no relevance by itself, otherwise this is just "fetch the latest N" |
| **R3 emotion match** | `arousal=1` and valence known | 60 with the same emotional leaning (gated too), and this turn's **depth budget +3** |
| **R4 standing** | every turn | **established** portraits only (`pending` is a guess without enough corroboration, never injected) + active summaries |
| **R5 raw drill-down** | **multi-dimensional only** | votes ≥ 3 **and** at least one strong dimension — raw text is the most expensive thing; neither "one dimension" nor "two weak ones" qualifies |

Ranking uses **four keys**: **votes → cue layer → freshness → centrality**. Votes are per-dimension and
weighted — semantic / entity / literal hits are **2 votes**, C2–C7 are **1 vote** — so an old memory hit
by several dimensions beats a fresh one, and ties break on freshness. The top N are injected (2 by
default, +3 on an emotional hit); **the rest go to the suppression list** together with "how far short
of the last selected one it fell", all into the trace — *why this surfaced and why that was held back is
part of the output*.

(Every threshold / budget above is a default in `core/config.py` — knobs live in one place; the cues and
actions are judged in `core/recall.py`, each with a comment on why it is the way it is.)

### Distillation (write): when to write, and how deep

**When extraction fires** (any of four): topic switch (**the main path** — cutting a scene and firing
extraction are the same event) · M turns without a switch · window over the token budget (only the
oldest batch is compressed, the tail stays verbatim) · session end. Between sessions a **health check**
picks up the slack (volume- or time-triggered: capped distillation + portrait review).

**One extraction yields two things**: the digest that goes back into the window (replacing the verbatim
text it compressed) and a scene card that goes into long-term memory — **compression *is* extraction**,
not two separate mechanisms. The unit is a **topic segment**, not a turn (a single sentence is too
fragmented to be a scene). A segment with nothing in it **keeps its digest but writes no card**: skipping
should mean "not into long-term memory", never "as if it never happened".

| Step | From → to | Nature |
|---|---|---|
| Distil 1 | S0 raw → S1 scene card | structured, **draws no conclusion**; drops raw text, entity index, open loops → memos on the way |
| Distil 2 | S1 → S2 topical summary | **aggregation** (reversible), **draws no conclusion** |
| Distil 3 | S2 → S3 portrait | **abstraction** (irreversible), **draws a conclusion** |

Only the concluding layer must clear **four gates**: **≥3 corroborations in the same situation**
(same `trigger_class` + semantically close triggers — the same reaction in a different situation is not
convergence) · `sources` required (traceability) · cross-source counting (only what *you* said counts as
your own expression) · long uncorroborated → demoted back to `pending`. After it stands, **revision**
keeps it honest: new evidence yields holds / revise / overturn; on revise the old record gets
`invalidated_at` and goes to history while a new one starts — which is how the dual timestamps can answer
"what did you look like to her that month". **A human veto really deletes.**

**Editing / regenerating only touches the window**: editing a line or regenerating a reply undoes that
turn (and everything after it) in the window and says it again — the undone turns **never reached
long-term memory**, so undoing them is the same as never having said them. The converse is just as hard:
**a stretch that has already been extracted is never rewritten** (in the UI it offers only copy and
read-aloud). So remember one thing: **memory is a snapshot, not a view** — what lands at extraction time
is "what was really said then", and later edits in the window never write back to it. To change
something already in long-term memory, you change the **understanding** (the Scenes page: edit or
archive a scene, summary or portrait — edits leave a trace); the raw text (S0) **cannot be changed**.
To void it, archive it or really delete it (real deletion is human-initiated only).

### Open loops · her hands · traces

- **Open loops** (memos): unfinished matters, including her own promises — **due → injected once (the
  injection is the accounting) → hit judged (updated / closed) → retired**. "Due" only means "eligible
  for injection"; whether to raise it is a matter of wording in the prompt. She **never speaks first** —
  her words only happen after yours.
- **Her hands** (tool belt): 5 tools, three permission tiers (autonomous / read-only / needs
  confirmation); the red line is **append-only** (deny, never allow); every call leaves a trace.
- **Traces**: one JSONL line per recall, one record per tool call, one per distillation run — the
  workbench is just these rendered.
- **All relations are derived**: no edge table — references are `sources`, adjacency is topic + time,
  raw text is the per-day doc, "when it was cited" is `profiles.evidence_at`.

All five stages (minimal loop / portraits & corroboration / memos & mirror / drift detection /
dialogue layer & workbench) are done.

## Three hard constraints

1. **No silent loss** — no path (crash / LLM failure / missing fields) silently deletes a scene that
   was already written; a corrupt DB is backed up and reported, never rebuilt; archive only sets a
   flag (reversible), ageing only changes status; **only a human click really deletes**.
2. **Degrade, don't crash** — LLM failures return conservative defaults, missing embeddings fall back
   to character overlap: coarser, not mute.
3. **S0 is outside automatic recall** — raw text has no index; automatic recall reaches it only by
   drilling down from a scene id (salvage is the one exception, and it is human-initiated).

## Run it

```bash
python -m core.dashboard        # the memory workbench (start here): chat left, watch recall right
python demo.py                  # stage 1: dialogue → remembered → recalled (prints the reasoning chain)
python demo_distill.py          # stage 2: aggregate → abstract → corroborate → revise → age → veto
python demo_memo.py             # stage 3: due → injected once → hit judged → retired
python demo_trend.py            # stage 4: drift detection (numbers moved ≠ the person changed)
python run_experiment.py --script your_script.json --fresh   # dialogue replay + report
python -m unittest discover -s tests                        # 629 tests, offline (mock LLM + embeddings)
```

- **The workbench**: chat on the left; on the right, this turn's cues, actions, recall (with the
  *why*) and suppression list; tabs below inspect the store (scenes / summaries / portraits /
  memos / entities / topics / salvage / traces). Persona and voice switch in the top bar.
- **Windows entry points**: `run.cmd` (console) · `start.vbs` (no console window, voice included) ·
  `switch.vbs` (desktop on/off switch). The Chinese-named `启动.vbs` / `开关.hta` are the originals;
  the ASCII names are aliases.
- `run_experiment.py` needs a script of **your own** dialogue material — the author's is private and
  not shipped (the format is documented at the top of that file).

### Using a real model

```bash
export AIR_LINK_LLM_ENDPOINT=https://api.example.com/v1
export AIR_LINK_LLM_API_KEY=sk-xxx
export AIR_LINK_LLM_MODEL=your-model
export AIR_LINK_EMBEDDING_ENDPOINT=https://api.example.com/v1   # or local://<model> for a local model
export AIR_LINK_EMBEDDING_API_KEY=sk-xxx
export AIR_LINK_EMBEDDING_MODEL=text-embedding-3-small
python demo.py --real
```

Or copy [`config.example.json`](config.example.json) to `config.local.json` (the Settings page writes
the same file). Precedence: defaults < `AIR2_*` < `config.local.json` < `AIR_LINK_*`; an empty key
field means "don't change it". ⚠️ **DeepSeek has no embeddings endpoint** — point embedding at a local
Ollama or OpenAI.

## Layout

```
air-link-01/
├── core/            the system, 24 modules (all but `__init__.py` carry a "module quick-ref" block)
├── web/index.html   the whole front end in one file (zero deps, CN/EN dictionary inline)
├── self/personas/   persona files (air / mia / xina) — files are the source, the UI is an editor
├── tests/           offline suite (mock LLM + mock embeddings)
├── tools/           pre-publish checks: secrets/PII scan, i18n coverage, doc sync (see tools/README.md)
├── docs/modules.md  generated module index: layer / upstream / downstream / entry points / boundary
├── tts/             optional voice service — only the **interface contract** ships (impl stays local)
├── data/            runtime data (DB / raw text / traces / snapshots) — never in git
└── ref/             local clones of reference projects (read-only, not imported) — never in git
```

## Docs

| File | When to read it |
|---|---|
| [`设计与实现对照-结构.md`](设计与实现对照-结构.md) | **before touching code**: layering + the nine steps + module → key functions (Chinese) |
| [`设计与实现对照-核对.md`](设计与实现对照-核对.md) | to audit "is this layer built, and built right" (per-layer checklists, Chinese) |
| [`docs/modules.md`](docs/modules.md) | to work on one module: layer / upstream / downstream / entry points / boundary + public API |
| [`参考与致谢.md`](参考与致谢.md) | what ideas were borrowed (no code was copied) |

The long-form design notes (six documents plus decision records) are **not distributed** — so a
reference like "设计稿 §x" in comments and docs is a *text coordinate* into the author's local notes,
not a link. Implementation and description drift apart? **Fix the description first, then the code.**

## Deliberately not built

Vector ANN index (exhaustive cosine over a few thousand rows is enough) · drift curves (drift is
traced, not plotted) · multi-user isolation (the design assumes a single user).

## License

**MIT** — see [`LICENSE`](LICENSE); code and in-repo docs share it.

**Not shipped** (each under its own upstream license): the **local TTS implementation**
(`tts/server.py` · `tts/voice.json` · `tts/requirements.txt` — only the interface contract in
[`tts/README.md`](tts/README.md) ships), TTS model weights (`tts/models/`), voice reference audio
(`tts/refs/`), local config `config.local.json` (API keys) and all runtime data under `data/`,
and the third-party clones under `ref/`.
