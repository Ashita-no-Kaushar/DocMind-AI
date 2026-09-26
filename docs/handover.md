# Handover — end of session 1

Read this first tomorrow. It records where the project actually stands, what is
uncommitted, and the order I would work in next.

## Repo state

- Path: `C:\Users\Kaushar\Downloads\DocMind-AI-main\DocMind-AI-main`
- Last pushed commit: `a75f42f` on `main`. **Everything since then is uncommitted.**
- Working tree is large and dirty by design. Nothing has been staged or committed.
- **A GitHub PAT was pasted into chat earlier and must be revoked.** It was never written to
  a file, but treat it as compromised.

## Current gate (all green as of the last run)

| Check | Result |
|---|---|
| `unittest discover -s tests` | 482 tests, 481 pass, 1 opt-in live-R2R skip |
| `tests.test_e2e_integration` | 18 pass, but **one flaky test, ~1 in 10** (see below) |
| `tests.test_e2e_contract` | 3 pass |
| `eval_harness.py --mock` | 42/42, 1 real-LLM skip |
| `ruff check .` / `black --check .` | pass, 61 files |
| `compileall`, `pip check`, `pipenv verify` | pass |
| `python -m research.map_ablation --check` | artifacts match code |
| Docker / live providers | unavailable here; no live result claimed |

## The one known failure — do not claim it is fixed

`tests.test_e2e_integration.BrowserProviderTests.test_real_browser_streams_from_threaded_fake_openai_provider`
fails roughly 1 run in 10, **including in isolation**. Evidence already collected:

- The browser-storage restore completes and re-persists the injected provider correctly —
  verified by reading `localStorage['docmind:settings']` inside `localStorage`, which contains
  `llm_backend = "LM Studio (Local AI)"` and all injected endpoints.
- Yet the Settings tab's provider selectbox renders `Ollama`.
- Switching to another tab and back does **not** correct it, so it is not a first-paint effect.
- No page errors, no Streamlit exception, the child process owns its port for the whole test.

This is an **application-side display defect**, tracked in `docs/todo.md` under
"Known defect: Settings provider display after restore". Next step: instrument
`utils/browser_settings.restore_settings_from_browser_storage` →
`st.selectbox(key="llm_backend")` in `components/tabs/settings.py` to find which run paints
the stale value, then fix the app rather than the assertion.

Two genuine harness bugs were already found and fixed here, so do not re-chase them:
port-reuse race (stale Streamlit answering the health check) and an assertion that read the
widget before the restore landed.

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

1. Fix the Settings provider display defect and get the E2E suite to 18/18 repeatedly.
2. Commit everything, then decide the PAT.
3. Confirm the resolution default on a real corpus before changing the shipped default —
   do not change it on synthetic evidence.
4. Bootstrap confidence intervals. With 25 answerable questions one hit is 4 points and no
   current difference is statistically established.
5. Build a corpus that produces a real accuracy crossover (documents with hundreds of chunks,
   or genuinely multi-hop questions). The harness already scales to any size.
6. Live LLM-planner measurement so the upper-bound column becomes real.
7. Multimodal is the big one and is blocked: needs a PDF rasteriser (`pypdfium2` is a pure
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
