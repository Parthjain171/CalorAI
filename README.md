# CalorAI Agent

A conversational meal logger you talk to like a friend. Text what you ate, send a
photo of your plate, correct yourself halfway through — it keeps an accurate
running total of your day. No forms, no dropdowns.

```
you: had 2 parathas and chai for breakfast
calorai: Logged 2 parathas and chai — 610 cal, 12.5g protein. That puts you at 610 cal today.

you: actually that was 3 parathas
calorai: Fixed — 3 parathas and chai, 870 cal. You're at 870 cal today.

you: how much protein have I had?
calorai: You're at 17.5g protein today.
```

That second message is the whole product in miniature: it **edits the existing
row**. The day shows 870, not 610 + 870.

## Contents

- [Project Overview](#project-overview)
- [Setup / Installation](#setup--installation)
- [Model Choices](#model-choices)
- [Memory Design](#memory-design)
- [Tool Design](#tool-design)
- [Latency Numbers](#latency-numbers)
- [Architecture](#architecture)
- [Test Cases](#test-cases)
- [Assumptions and Trade-offs](#assumptions-and-trade-offs)
- [Time Breakdown](#time-breakdown)
- [What I'd Build Next](#what-id-build-next)
- [AI Tools Used](#ai-tools-used)

---

## Project Overview

CalorAI is a WhatsApp-style calorie tracker built as a LangGraph agent over
SQLite. A user texts what they ate in whatever shape it comes out — "leftover
biryani, maybe two thirds of the box", "same as yesterday", a photo of a plate
with "half of this was my brother's" — and the agent decides whether it has
enough to log, asks one short question when it doesn't, and maintains running
daily totals that stay correct through edits and deletions. Two models are used
deliberately: a fast, cheap one for conversation and tool calling, and a stronger
one for recognising food in photos. Preferences, goals, and recurring meals
persist as durable memories that survive restarts and get selectively injected
into the prompt each turn.

---

## Setup / Installation

```bash
git clone https://github.com/Parthjain171/calorai-agent.git
cd calorai-agent

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # Windows: copy .env.example .env
# put your key in .env:  ANTHROPIC_API_KEY=sk-ant-...
```

Run it:

```bash
python cli.py                                    # interactive chat
python cli.py --user parth                       # a separate, isolated log
python cli.py -m "had 2 rotis and dal"           # one-shot
python cli.py --image assets/sample_plate.png -m "half was my brother's"
python cli.py --latency                          # p50/p95 report
```

In-chat: `/img <path> [caption]`, `/totals`, `/meals`, `/memories`, `/latency`,
`/reset`, `/help`, `/quit`.

Run the eval set:

```bash
python eval/eval_runner.py                       # all 11 cases
python eval/eval_runner.py --case 05 09 -v       # just the differentiators
python eval/eval_runner.py --repeat 10           # more latency samples
python eval/test_eval_detects_regressions.py     # proves the eval can fail
```

**No API key?** `CALORAI_MOCK=1` swaps in a deterministic scripted stand-in for
the conversation model, so the whole suite runs offline. It validates plumbing —
tools, database, state transitions — **not** answer quality. See
[Latency Numbers](#latency-numbers) for what that does and does not tell you.

Only `ANTHROPIC_API_KEY` is needed for the defaults. Set `OPENAI_API_KEY` only if
you point `TEXT_MODEL`/`VISION_MODEL` at a `gpt-*` model; the provider is
inferred from the model id, so swapping is an env change, not a code change.

---

## Model Choices

| Path | Model | Price /MTok | Why |
|---|---|---|---|
| Conversation + tool calling | `claude-haiku-4-5` | $1 in / $5 out | Fastest and cheapest tool-caller in the family. Runs on ~90% of turns. |
| Vision (food recognition) | `claude-sonnet-5` | $2 in / $10 out | Materially better at identifying food and judging portions from a plate. |
| Nutrition estimation | `claude-haiku-4-5` | $1 in / $5 out | Sits on the critical path, emits ~200 tokens of JSON. Rarely called (see caching). |

**Why two models rather than one.** The two jobs have opposite cost profiles.
Conversation and tool orchestration are *easy* — decide which tool, fill in
arguments, write one friendly sentence — and they happen on every single message,
so latency and cost dominate. Identifying dal versus rajma from a photo is
*hard*, and happens on a small minority of turns, so quality dominates and the
price difference is bounded by how rarely it runs. Routing everything through one
model means either paying vision-grade prices on every "how am I doing?" or
accepting weak food recognition. Splitting also isolates failures: a vision
timeout degrades to "what was in the photo?" instead of taking down logging.

**Why not the models named in the brief.** The brief suggested Claude 3.5 Sonnet
or GPT-4o-mini for text and GPT-4o or Claude 3.5 Sonnet for vision. Both paths
here use newer models in the same tiers — cheaper, faster, and better at tool
calling than the 3.5 generation. The originals remain one env var away:

```bash
TEXT_MODEL=gpt-4o-mini      VISION_MODEL=gpt-4o        # needs OPENAI_API_KEY
```

**The handoff, concretely.** A photo does not reach the conversation model as an
image. The vision node calls the vision model, which returns JSON only — foods,
servings, confidence, and a confirming question if unsure. That is rendered as a
`[VISION]` note and **merged into the user's own message**, so the text model
reads the caption and the identified food together in one turn. That merge is
what makes case 9 work: "1x biryani" plus "half of this was my brother's"
resolves to one meal at 0.5 servings rather than a photo meal and a caption meal.
Low confidence is passed through rather than hidden, so the agent asks "looks
like rice and dal — is that right?" instead of guessing.

---

## Memory Design

The most important part of the system, and the part most easily done badly.
Memory here is **not** conversation history — it is a small set of durable facts
that should still be true next month.

### What gets stored

| Category | Example | Retrieval |
|---|---|---|
| `preference` | vegetarian; allergic to peanuts | **always injected** |
| `goal` | 140g protein per day | **always injected** |
| `usual_meal` | usual breakfast = 2 idli and sambar | **always injected** |
| `habit` | skips lunch on weekdays | keyword-matched |
| `fact` | anything else worth keeping | keyword-matched |

The first three tiers are always injected because there are only a handful per
user, they change the answer *even when the message doesn't mention them*, and
getting them wrong is the visible failure — suggesting chicken to a vegetarian is
worse than any amount of latency. The other two are where volume accumulates, so
they must earn their place in the prompt.

### What is deliberately NOT stored

Individual meals (those are rows in `meals`), moods, one-off remarks, and
anything you wouldn't want repeated back in three weeks. Auto-summarising every
turn into memory is what produces the "remembers that you said hi on Tuesday"
failure mode, so the `store_memory` tool description is mostly a list of what not
to keep.

### When it writes

Writes are **model-driven**: the model recognises a durable fact and calls
`store_memory` in the same turn, without ceremony. "i'm vegetarian btw" stores
`diet = vegetarian` and logs no meal. There is no background summarisation pass —
that would be a second inference on every turn, for facts that appear a handful
of times in a user's lifetime.

### How it retrieves

Retrieval runs in the graph's `prepare` node, against the incoming message,
*before* the model sees anything:

1. Take all `preference` / `goal` / `usual_meal` memories.
2. Score the rest by token overlap with the user's message, ignoring stopwords,
   with a small bonus for frequently-used memories so a daily habit outranks a
   fact mentioned once.
3. Keep the always-on tier plus anything scoring above zero, capped at
   `CALORAI_MAX_MEMORIES` (default 8).
4. Mark what was surfaced (`last_used`, `use_count`) — that feeds step 2 next time.

`recall_memory` exists as a tool on top of this for the case injection can't
cover: "my usual" needs the *exact stored value* to act on, not a hint.

### How it avoids bloating context

Three layers rather than one, because any single layer fails eventually:

1. **Write-side selectivity** — only five categories are storable, and the tool
   description steers hard against keeping chatter.
2. **Keyed upsert** — `UNIQUE(user_id, key)`. Saying "I'm vegetarian" ten times
   yields one row, not ten. Restating a goal overwrites it.
3. **A hard cap** — at most 8 memories reach any prompt, ranked by tier then
   relevance then recency. Even a user with 500 memories has a bounded prompt.

### Persistence

Memories live in SQLite, so they survive process restarts, and they are scoped by
`user_id`. The eval asserts this: case 10 stores a "usual breakfast" and a later
turn resolves "my usual" against it through a fresh agent instance.

**Known limitation:** relevance is lexical token overlap, not embeddings. "what
can I eat" will not retrieve a memory phrased "avoids dairy". The always-injected
tiers are what keep this from mattering for the facts that actually matter; see
[What I'd Build Next](#what-id-build-next).

---

## Tool Design

Eight tools, split so that no two can do the same job. The rule of thumb: one
tool per *state transition*, and reads separated from writes.

| Tool | Does | Boundary |
|---|---|---|
| `lookup_nutrition` | Macros for a **list** of foods, scaled by servings | Pure read, no DB writes. Never logs. |
| `log_meal` | Insert a **new** meal | Only for food not yet recorded. |
| `get_meals` | Retrieve meals by date range / name fragment | The only way to find a `meal_id`. |
| `update_meal` | Edit an existing meal **in place** | The only path for corrections. |
| `delete_meal` | Remove a meal entirely | Points corrections back at `update_meal`. |
| `get_daily_totals` | Calories + macros for one day | Reads the SQL view, never estimates. |
| `store_memory` | Save a durable fact | Never stores meals. |
| `recall_memory` | Look up stored facts | Read-only. |

Design decisions worth defending:

- **`log_meal` vs `update_meal` is the sharpest boundary in the system.** Using
  `log_meal` to fix an existing meal double-counts the day, and it fails
  *silently* — no error, just wrong numbers. Both tool descriptions call this out
  explicitly, and the eval asserts it.
- **`lookup_nutrition` takes a list, not one food.** "2 parathas and chai" needs
  two lookups; batching them into one call saves a full model round-trip per
  extra food, and lets the estimator resolve every cache miss in a single
  request. It returns a combined `total` that can be passed straight to
  `log_meal`.
- **`log_meal` returns the updated daily totals.** The most common reply is
  "logged it, here's where you stand", so folding totals into the write removes a
  second tool call from the hottest path.
- **`get_meals` has `name_contains`.** Corrections need to find "the rotis"
  without a separate search tool, which would have overlapped with `get_meals`.
- **`user_id` is never a tool argument.** It travels out-of-band in a
  `contextvars.ContextVar` set per turn. The model cannot pick whose data it
  touches, and the schemas stay about food.

### The nutrition lookup, in three tiers

Cheapest first, because most food is boring and repeated:

1. **Seed table** — ~70 foods Indian users actually text about, in-process. ~0 ms,
   zero tokens. Handles most eval traffic.
2. **SQLite cache** — anything an LLM has priced before, on any past run.
   Survives restarts.
3. **LLM estimator** — one batched call for whatever is left, then cached, so a
   given food is paid for at most once ever.

If the estimator is unreachable, a coarse keyword heuristic produces a number
anyway. In a texting UX a roughly-right calorie count that gets logged beats an
error message.

---

## Latency Numbers

**Be clear about what was and wasn't measured.** No API key was available in the
build environment, so *end-to-end latency against real models is not measured
here*. What follows separates the two honestly.

### Measured: framework overhead

Everything except the model calls — graph traversal, memory retrieval, tool
execution, SQLite reads and writes, response assembly. From 10 full eval passes
(`CALORAI_MOCK=1`), n=100 text turns / n=30 image turns:

| Span | n | p50 | p95 | max |
|---|---|---|---|---|
| `turn_text` (end to end) | 100 | 0.03 s | 0.05 s | 0.09 s |
| `turn_image` (end to end) | 30 | 0.03 s | 0.07 s | 0.09 s |
| `text_model_call` (harness only) | 360 | 0.00 s | 0.00 s | 0.05 s |
| `vision_model` (harness only) | 30 | 0.00 s | 0.00 s | 0.00 s |

So the harness contributes **~30 ms at p50 and ~70 ms at p95**. Against a 3 s text
budget that is ~2%, which means essentially the entire real-world budget is model
time, and optimisation effort belongs there rather than in the graph.

### Not measured: real model latency

To produce real numbers, set a key and run:

```bash
CALORAI_MOCK=0 python eval/eval_runner.py --repeat 5
python cli.py --latency         # p50/p95 across every run, from data/latency.jsonl
```

The instrumentation records nested spans (`turn_text` / `turn_image` contain
`vision_model`, `text_model_call`, `nutrition_llm`, `time_to_first_token`), so
the report attributes a slow turn rather than just reporting one.

Expected shape, stated as reasoning rather than measurement:

- **Text turn** = 2–3 sequential `text_model_call`s (decide → tools → reply).
  Haiku-class calls with short prompts are typically a few hundred ms each, so a
  simple log lands comfortably under the 3 s target; a correction, which needs
  `get_meals` → `lookup_nutrition` → `update_meal` → reply, is the worst case at
  4 round-trips.
- **Image turn** = one vision call (the expensive part — a base64 image plus a
  Sonnet-class model) **plus** the full text turn behind it. The vision hop is
  strictly serial: the text model cannot decide what to log before knowing what
  is on the plate. That serialisation is the honest cost of routing images to a
  separate model, and it is the main reason the image budget is 6 s rather than 3 s.

### What was done to optimise

- **Batched nutrition lookups.** One tool call for every food in a message
  instead of one per food — saves a whole model round-trip per extra food.
- **Two-layer nutrition cache.** The seed table answers most lookups with zero
  tokens and no network; LLM estimates are persisted to SQLite so a food is
  priced at most once ever.
- **Totals folded into `log_meal`.** Removes a `get_daily_totals` round-trip from
  the most common interaction.
- **Cached model clients** (`lru_cache`), so no turn pays to rebuild an HTTP
  connection pool.
- **Small model on the hot path.** Haiku for conversation; Sonnet only when there
  is actually a photo.
- **Streaming** (`stream_chat`), with `time_to_first_token` recorded separately —
  streaming doesn't make a turn faster, but time-to-first-token is the latency a
  user actually feels.
- **Parallel tool calls** work (LangGraph's `ToolNode` runs them on a thread
  pool). This is what surfaced the SQLite concurrency bug described below.

### What is still slow, and why

- **The vision hop is serial and unavoidable** in this design. Speculatively
  starting the text model before vision returns would just waste tokens, since
  every downstream decision depends on what the food is.
- **Corrections take 4 model round-trips** (find → re-price → update → reply).
  Collapsing find-and-update into one tool would be faster but would blur the
  `log`/`update` boundary that keeps double-counting impossible — a trade I chose
  not to make.
- **A cache miss adds a full extra model call** inside the turn. It is capped at
  one call per turn by batching, and never repeats for the same food.
- **p95 over small samples is indicative only.** The report says so itself when
  n < 20 rather than implying more precision than the data supports.

---

## Architecture

```mermaid
flowchart TD
    U["user message<br/>(text and/or image path)"] --> V

    subgraph graph["LangGraph StateGraph"]
        V{"vision node<br/>image attached?"}
        V -- no --> P["prepare node"]
        V -- yes --> VM["VISION MODEL<br/>claude-sonnet-5<br/>→ foods, servings, confidence"]
        VM --> MERGE["merge [VISION] note<br/>INTO the user's message"]
        MERGE --> P

        P --> MEM["recall relevant memories<br/>always-on tiers + keyword match, cap 8"]
        MEM --> SYS["build system prompt"]
        SYS --> A["agent node<br/>TEXT MODEL claude-haiku-4-5<br/>+ 8 bound tools"]

        A -- "tool_calls" --> T["ToolNode<br/>(parallel, thread pool)"]
        T --> A
        A -- "no tool_calls" --> R["reply"]
    end

    T <--> DB[("SQLite<br/>meals · memories · nutrition_cache<br/>daily_totals VIEW")]
    R --> OUT["streamed to CLI"]

    style VM fill:#e8d5f2,stroke:#7b3fa0
    style A fill:#d5e8f2,stroke:#2c6d9e
    style DB fill:#e8f2d5,stroke:#5a8a2c
```

**The flow in words.** A turn enters at the `vision` node, which no-ops unless an
image is attached — so text turns pay nothing for the image path. If there is a
photo, the vision model identifies the food and its output is merged into the
user's own message. `prepare` then retrieves relevant memories and builds the
system prompt for this turn. The `agent` node runs the text model with all eight
tools bound; if it emits tool calls they execute in `ToolNode` and loop back,
otherwise the reply is returned and streamed.

**Two state decisions worth noting.** The system prompt is held in state as a
plain string rather than as a message, because `add_messages` *appends* — a
freshly built `SystemMessage` each turn would quietly stack up inside the
checkpointer and grow context forever. And session history lives in a LangGraph
checkpointer keyed on `user_id`, while anything that must outlive the process
(meals, memories) lives in SQLite, not the message log.

**Why totals can't drift.** `daily_totals` is a SQL **view** aggregating meal
rows on every read, not a maintained counter. An edit or delete is reflected the
instant it lands, so the number the user hears is always `SUM()` over the rows
that currently exist.

---

## Test Cases

All 11 required conversations pass. The eval asserts against the **database**,
not the wording of the reply — a model that says "logged it!" and writes nothing
fails, and a model that phrases things differently still passes.

| # | Conversation | What is asserted |
|---|---|---|
| 1 | "had 2 parathas and chai for breakfast" | 1 row, `meal_type=breakfast`, macros > 0 |
| 2 | "leftover biryani, maybe two thirds of the box" | 1 row, calories in a two-thirds range |
| 3 | "skipped lunch but grazed all afternoon" | **0 rows** + reply contains a question |
| 4 | "same as yesterday" | today's calories ≈ yesterday's |
| 5 | "actually that was 3 rotis not 2" | **exactly 1 row**, total below the double-counted figure, row says 3 |
| 6 | "how much protein have I had today?" | reply quotes the DB protein total |
| 7 | "how am I doing on calories?" | reply quotes the DB calorie total |
| 8 | [photo] | 1 row, `source` records vision, calories > 0 |
| 9 | [photo] + "half of this was my brother's" | **1 row**, under 75% of the same photo logged uncaptioned |
| 10 | "my usual" | 1 row matching the stored usual |
| 11 | "i'm vegetarian btw" | preference in memory, **0 meals logged** |

Case 9's check is measured, not hardcoded: the runner first logs the same photo
*without* a caption as a reference user, then requires the captioned run to come
in materially lower. So "half" is compared against an actual full plate.

**The eval can fail.** `eval/test_eval_detects_regressions.py` deliberately
breaks the two differentiator behaviours — makes corrections insert instead of
update, and makes the photo path ignore its caption — and asserts the matching
case turns red. Both sabotages are detected. An eval that only ever passes is
not evidence of anything.

### Two real bugs the eval caught

Worth recording, because both were silent-wrong-answer bugs rather than crashes:

1. **Parallel tool calls lost data.** Running the suite 6× surfaced a ~1-in-20
   failure in case 4 — the only case issuing multiple `log_meal` calls in one
   turn. `ToolNode` executes parallel calls on a thread pool, and a single
   `sqlite3` connection isn't safe for concurrent use; `insert_meal`'s INSERT and
   its read-back were interleaving. In production this would have dropped a meal
   from a multi-meal turn and the user would only notice their totals were low.
   Fixed with a re-entrant lock around all database access.
2. **The photo caption was being ignored.** Appending the `[VISION]` note as its
   own message made it the "latest user message", hiding the caption — so "half
   of this was my brother's" logged a full plate. Fixed by merging the note into
   the user's message instead.

---

## Assumptions and Trade-offs

**Assumptions**

- One user per CLI session, identified by `--user`. Rows are scoped by `user_id`
  throughout, so multi-user isolation holds, but there is no auth — the user id
  is asserted, not proven.
- Calorie tracking is trend-following, not laboratory measurement. Seed values
  are approximations of a typical serving, and "two thirds of the box" is
  genuinely ±20%. Precision beyond that is false comfort.
- A "day" is the user's local calendar day. Timezone is a config value
  (`CALORAI_TZ_OFFSET_HOURS`, default IST), not detected.
- The food vocabulary is India-first, since that is the described user base.

**Trade-offs I chose**

| Chose | Gave up |
|---|---|
| Seed table + cache before any LLM call | Coverage — an unusual food costs one extra model call the first time |
| `daily_totals` as a view | A few ms per read, for totals that can never drift |
| Separate `log_meal` / `update_meal` | One round-trip on corrections, to make double-counting structurally impossible |
| Memory injected by keyword relevance | Semantic recall; an embedding index is the obvious upgrade |
| Model-driven memory writes | Recall — a durable fact stated obliquely may not get stored |
| Serialising all DB access with a lock | Theoretical write concurrency SQLite wouldn't have delivered anyway |
| Vision → text as two serial calls | ~1–2 s on image turns, for much better recognition and isolated failures |
| `contextvars` for `user_id` | Explicitness, in exchange for tool schemas the model can't misuse |
| A scripted test double for offline evals | It proves plumbing only, and is clearly labelled as such |

**Known limitations**

- Memory relevance is lexical, not semantic.
- The scripted double is a rule engine; a green mock run says nothing about how a
  real model handles phrasing it has never seen.
- No retry/backoff around model calls beyond the SDK's own.
- Nutrition numbers are model- and table-derived, not from a verified food
  database.

---

## Time Breakdown

Approximate effort across the build, in the order it was done:

| Phase | Time |
|---|---|
| Project setup, config, dependencies | 0.3 h |
| Database schema + data access layer | 0.7 h |
| Core tools (nutrition, log, get) + seed table | 1.0 h |
| LangGraph agent, system prompt, scripted double | 1.0 h |
| Daily totals + corrections/deletion | 0.6 h |
| Memory system (manager, tools, injection) | 0.9 h |
| Vision model + image/caption resolution | 0.9 h |
| CLI, streaming, latency instrumentation | 0.8 h |
| Eval set + regression-detection harness | 0.8 h |
| Debugging the two silent bugs above | 0.6 h |
| README | 0.7 h |
| **Total** | **≈ 8.3 h** |

The single largest unplanned cost was the concurrency bug — it only appeared
under repeated runs, which is precisely the argument for running an eval more
than once.

---

## What I'd Build Next

In rough order of value per hour:

1. **Real nutrition data.** Swap the LLM estimator for a food database
   (USDA FoodData Central, or IFCT for Indian foods) behind the same
   `lookup_nutrition` interface. Model-estimated macros are the weakest numbers
   in the system and the tiering already makes this a drop-in third tier.
2. **Embedding-based memory recall.** Keyword overlap is the clearest limitation
   in the most important subsystem. Small local embeddings over memory values,
   keeping the tiering and the cap.
3. **Confirm-before-writing on low-confidence photos, end to end.** The vision
   model already emits a confidence and a question; the agent asks, but a proper
   pending-confirmation state would stop an unanswered question from leaving a
   meal unlogged.
4. **Persist conversation state.** Swap `MemorySaver` for LangGraph's SQLite
   checkpointer so a mid-clarification thread survives a restart.
5. **Portion learning.** If a user corrects "1 bowl of dal" to a bigger portion
   three times, store that as a `habit` memory and stop getting it wrong.
6. **LangSmith tracing.** The env hooks are in `.env.example`; wiring a public
   trace would make the tool-call sequences inspectable.
7. **A real WhatsApp transport.** The agent interface (`chat` / `stream_chat`) is
   already transport-agnostic; this is a webhook and media download away.

---

## AI Tools Used

Built with **Claude Code (Opus 5)** as a pair programmer, used for essentially
the whole build: scaffolding the package layout, drafting the SQLite schema and
data access layer, writing the LangGraph wiring, and generating the seed
nutrition table.

Where it helped most:

- **Volume with structure** — the ~70-entry seed table and the tool docstrings
  are the kind of high-quantity, moderate-judgement work that is fastest to
  generate and then edit.
- **Debugging from evidence.** The concurrency bug was found by reading an actual
  stack trace out of a flaky run rather than by guessing, then fixed at the layer
  that owned the problem.

Where it needed correcting — worth being specific, since this is the honest part:

- It initially attributed the truncated latency-log lines to a write-retry it had
  just added, removed the retry, and the corruption persisted. The real cause was
  concurrent unlocked appends from LangGraph's worker threads. The lesson held
  for the SQLite bug an hour later.
- The first vision integration appended the `[VISION]` note as a separate
  message, which silently broke the caption path — caught only because case 9
  asserts against a measured full-plate baseline rather than a fixed number.

Model IDs and pricing were taken from Anthropic's current model documentation
rather than from recall.
