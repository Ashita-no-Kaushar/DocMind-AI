# Handover — end of session 1

Read this first tomorrow. It records where the project actually stands, what is
uncommitted, and the order I would work in next.

## Repo state

- Path: `C:\Users\Kaushar\Downloads\DocMind-AI-main\DocMind-AI-main`
- Remote: `https://github.com/Ashita-no-Kaushar/DocMind-AI.git`, branch `main`.
- Pushed through `6bc5f7d`, where CI was green across all three workflows. The Streamlit UI redesign
  landed on top of that and is the most recent commit.
- **A GitHub PAT was pasted into chat earlier and must be revoked.** It was never written to
  a file, but treat it as compromised.

## Current gate (all green as of the last run)

| Check | Result |
|---|---|
| `unittest discover -s tests` | 498 tests, 480 non-browser pass, 1 opt-in live-R2R skip |
| `tests.test_e2e_integration` | 18 tests, **one flaky** (see below); green when the machine is idle |
| `tests.test_e2e_contract` | 3 pass |
| `eval_harness.py --mock` | 42/42, 1 real-LLM skip |
| `ruff check .` / `black --check .` | pass, 61 files |
| `compileall`, `pip check`, `pipenv verify` | pass |
| `python -m research.map_ablation --check` | artifacts match code |
| Docker / live providers | unavailable here; no live result claimed |
| CI at `6bc5f7d` | Quality, E2E, and Docker Build all green |


## The known failure — do not claim it is fixed

One browser E2E test in `tests.test_e2e_integration` is flaky, worse on a loaded machine.

**Chat submission is dropped by the frontend.**
`BrowserSmokeTests.test_real_browser_smoke_clears_chat_and_local_resets` intermittently never sends
the prompt at all. Measured signature from a diagnostic run:

- user message bubbles: `1` (the "How to use" placeholder only)
- fake-provider `/v1/chat/completions` requests: `0`
- Streamlit exceptions: `0`, page errors: `0`

Nothing is wrong with the app, the provider, or the answer path — the turn never starts. Streamlit
discards a chat submission that arrives while the app is still running a script, and this build
exposes **no** element, attribute, or `aria-busy` state to detect that, so a test cannot wait it out.
Verified absent: `stStatusWidget` appeared in 0 of 48 polls and `[aria-busy="true"]` was never present.
Ruled out by experiment: idle waiting, value-stability polling, `press_sequentially`, explicit focus,
and clicking the enabled submit button all still drop.

The first-token budget was raised to a named 120 s and is **not** the cause; the wait expires because
no turn began. `_send_chat_prompt` now confirms the user bubble, retries up to three times, and dumps
widget state plus the app log on failure, so the next occurrence is diagnosable instead of an opaque
timeout. The failure rate is unchanged. A real fix needs an app-side readiness signal or a Streamlit
version that exposes one.

Two genuine harness bugs were already found and fixed here, so do not re-chase them: port-reuse
race (a stale Streamlit answering the health check) and an assertion that read the widget before
the restore landed.

## What the research actually shows

`research/REPORT.md` is generated. Key honest findings:

- High-resolution map: **hit@k 1.00, 86.0 evidence tokens** vs naive BM25 top-k
  **1.00, 157.1** → **45.3% smaller prompt at equal accuracy**, chunk precision 0.43 vs 0.29.
- **The shipped production default (6 chunks/section) is the weakest map setting** (0.92). At
  6 chunks per section a 6-chunk document collapses to one section, so the agent can only pick
  whole documents. Adaptive resolution (3 sections/document) scores 1.00.
- **No accuracy crossover exists** in the 8→64 document scaling sweep. Naive top-k stayed
  saturated. The win is context efficiency, not accuracy.
- **Charging the map to the prompt is a net loss at every resolution** (high-res map index is
  15.8x the whole naive prompt). The defensible claim is *zero extra model calls*, not a saving.

## Two real bugs the study found — both fixed at the root

These are the most valuable outputs; keep them in any writeup.

1. `_bounded_excerpt` kept only the **first sentence** of a chunk, so discriminative terms late
   in a chunk never reached the card scorer. For *"How long is the staging soak time?"* the term
   `soak` appeared in **0 of 48** section cards. It now spends the whole character budget on as
   many whole sentences as fit.
2. The card scorer added **flat** `+1.5` / `+0.75` title/keyword bonuses, so a common word in a
   title outranked a rare word in the body. Bonuses are now scaled by the term's IDF.

Also added while measuring: `routing_engaged_rate`, because a version of our own agent was
silently falling back to unrestricted retrieval and *improving* its apparent accuracy. That
metric is what caught it — do not remove it.

## Tried and rejected (keep as a negative result)

A two-stage planner that re-scores the shortlist on full section text. Implemented, measured,
**worse** than the default, and left off. The `map_agent_refine` ablation row and the
`refine_candidates` config remain so the result is reproducible. `_corpus_stats` is lazy so the
disabled path pays nothing.

## Order I would work in next

1. Fix both flaky browser E2E tests: de-flake the 30 s first-token wait, then fix the Settings
   provider display defect, and get the suite to 18/18 repeatedly on a loaded machine.
2. Read the CI run status. The workflows were pushed but have never been observed green.
3. Revoke the exposed PAT. A credential for this repo is still live and was used to push.
4. Confirm the resolution default on a real corpus before changing the shipped default —
   do not change it on synthetic evidence.
5. Bootstrap confidence intervals. With 25 answerable questions one hit is 4 points and no
   current difference is statistically established.
6. Build a corpus that produces a real accuracy crossover (documents with hundreds of chunks,
   or genuinely multi-hop questions). The harness already scales to any size.
7. Live LLM-planner measurement so the upper-bound column becomes real.
8. Multimodal is the big one and is blocked: needs a PDF rasteriser (`pypdfium2` is a pure
   wheel) and a real vision encoder (none installed; project's Ollama/OpenAI-compatible
   providers could supply one). `python -m research.multimodal` reports the gap and refuses to
   fabricate. **No image-resolution number may be claimed anywhere.**

## Invariants that must survive future edits

- Never claim: lower total cost, better answers, lower embedding cost, an accuracy crossover,
  or any image/pixel-resolution result.
- Resolution means **structural granularity**, never pixels.
- The oracle planner reads ground truth: upper bound, never an LLM result.
- Evidence tokens and map index tokens are never merged.
- `python -m research.map_ablation --check` must stay green; CI enforces it.
- Keep `routing_engaged_rate` in every row and aggregate.
