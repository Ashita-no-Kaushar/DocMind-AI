# DocMind AI — Full Project Evaluation Report

_Generated 2026-09-26 14:32 — 42/42 scored checks passed, 1 skipped, raw score **100.0%**_

This harness uses small local fixtures. It is not a browser test, a production load test, or proof that generated answers are always factually grounded.

## Summary

| Suite | Passed | Skipped | Score | Scope |
|---|---:|---:|---:|---|
| ingestion | 7/7 | 0 | **100.0%** | Loader exclusions, selected formats, chunking, dedup, titles, cache, helper validation |
| retrieval | 12/12 | 0 | **100.0%** | Fixture retrieval, query cleanup, hybrid ranking, and no-match rejection |
| generation | 5/5 | 1 | **100.0%** | Prompt guards, mocked no-match path, tone prompts, history, three real LLM trials |
| performance | 5/5 | 0 | **100.0%** | Small-fixture timings, cache round-trip, Eco settings, batch shrinking |
| robustness | 6/6 | 0 | **100.0%** | Empty/binary handling, URL/GitHub validation, token cleanup |
| architecture | 7/7 | 0 | **100.0%** | Provider definitions, export, persistence contract, mocked R2R health |
| **Overall** | **42/42** | **1** | **100.0%** | Raw scored-check rate |

## Retrieval deep-dive (plain vs Hybrid)

_Embedding: `hash-mock` — hit = expected doc in top-3, correct rejection = no-match returns 0 nodes_

| Metric | Plain RAG | DocMind Hybrid |
|---|---:|---:|
| Factual hit rate | 80% | **100%** |
| Correct rejection | 50% | **100%** |
| Avg latency | 0 ms | 1 ms |

| # | Query | Expected | Feature | Plain | Hybrid | Plain top | Hybrid top |
|---|---|---|---|:---:|:---:|---|---|
| 1 | What is the refund policy? | refund_policy.txt | basic | ✓ | ✓ | refund_policy.txt | refund_policy.txt |
| 2 | money back rules | refund_policy.txt | synonym | ✗ | ✓ | — | hr_policy.txt |
| 3 | 30-day returns | refund_policy.txt | hyphen | ✓ | ✓ | refund_policy.txt | refund_policy.txt |
| 4 | refund policy batao | refund_policy.txt | hinglish | ✓ | ✓ | refund_policy.txt | refund_policy.txt |
| 5 | annual report revenue growth | annual_report.txt | title-aware | ✓ | ✓ | annual_report.txt | annual_report.txt |
| 6 | How to fetch a URL in Python? | python_guide.txt | basic | ✓ | ✓ | python_guide.txt | python_guide.txt |
| 7 | How many leave days per year? | hr_policy.txt | basic | ✓ | ✓ | hr_policy.txt | hr_policy.txt |
| 8 | pasta cooking time | cooking.txt | basic | ✓ | ✓ | cooking.txt | cooking.txt |
| 9 | What is the capital of France? | — (no match) | no-match | ✗ | ✓ | refund_policy.txt | — |
| 10 | quantum physics equations | — (no match) | no-match | ✓ | ✓ | — | — |
| 11 | refunded purchases | refund_policy.txt | stemming | ✓ | ✓ | refund_policy.txt | refund_policy.txt |
| 12 | tell me about the annual report | annual_report.txt | filler-filter | ✗ | ✓ | — | annual_report.txt |

## Suite: ingestion (7/7 scored, 0 skipped — 100.0%)

| Test | Pass | Detail |
|---|:---:|---|
| excluded_patterns | PASS | loaded=1 patterns=34 355ms |
| title_aware | PASS | title='Annual Report' 0ms |
| dedupe | PASS | 3 -> 2 269ms |
| multiformat_index | PASS | docs=4 docx=yes 305ms |
| cache_key_and_persist | PASS | key=f7756eff4949b3aa5378 persist=hit 215ms |
| min_chunk_filter | PASS | MIN_CHARS=15 docs_in=2 65ms |
| helpers_github_normalize | PASS | owner/repo=True url=True reject_gitlab=True 6ms |

## Suite: generation (5/5 scored, 1 skipped — 100.0%)

| Test | Pass | Detail |
|---|:---:|---|
| qa_template_guards | PASS | has_guards=True len=938 0ms |
| query_helpers | PASS | hyphen=True hinglish=True synonym=True 0ms |
| no_match_fallback | PASS | fallback=hit 30ms |
| tone_presets | PASS | presets=6 distinct=True 6ms |
| multi_turn_history | PASS | messages_in_prompt=4 15ms |
| e2e_real_llm | SKIP | skipped (mock mode) 0ms |

## Suite: performance (5/5 scored, 0 skipped — 100.0%)

| Test | Pass | Detail |
|---|:---:|---|
| ingest_5_docs | PASS | 0.1s for 5 docs 107ms |
| cache_round_trip | PASS | cold 0.0s vs cache load 0.02s 121ms |
| retrieval_latency | PASS | vector 0ms hybrid 1ms 71ms |
| eco_mode_trims | PASS | ctx 4800->3200 pred 512->256 batch 16->4 0ms |
| batch_oom_resilience | PASS | shrunk to 4 after OOM 0ms |

## Suite: robustness (6/6 scored, 0 skipped — 100.0%)

| Test | Pass | Detail |
|---|:---:|---|
| github_validation | PASS | 5/5 cases |
| url_validation | PASS | https=True bad=True xss=True |
| empty_docs_rejected | PASS | Index creation failed safely. |
| binary_exclusion | PASS | loaded 1 (png excluded) |
| special_chars_tokenization | PASS | tokens=['30', 'day', 'money', 'back', 'refund'] |
| hinglish_filler_filter | PASS | rewritten='refund polici plea' |

## Suite: architecture (7/7 scored, 0 skipped — 100.0%)

| Test | Pass | Detail |
|---|:---:|---|
| backend_preset_definitions | PASS | definitions=['Ollama', 'OpenAI', 'LM Studio (Local AI)', 'TabbyAPI', 'OpenAI-compatible'] |
| export_docx | PASS | docx 36712 bytes |
| browser_settings_contract | PASS | settings=39 secrets_excluded=True |
| ollama_helpers | PASS | estimate=2 trim=1 |
| embedding_verify | PASS | mock verify |
| r2r_health | PASS | health mocked |
| no_torch_at_import | PASS | torch imports=0 |

## Implemented Retrieval Features

- Vector candidates are fused with a CPU BM25 ranking through Reciprocal Rank Fusion.
- With a positive similarity cutoff, weak vector candidates require positive BM25 evidence.
- Short queries can receive curated BM25 synonym tokens; the vector query remains separate.
- Hyphens are split, basic number words are normalized, and selected filler words are removed.
- File-backed chunks receive a title prefix; website documents use source URLs without file titles.
- A stemmed-token overlap filter removes near-duplicate chunks before embedding.
- Up to five persisted index directories are retained when cache writes are available.
- Eco Mode lowers configured embedding batch, output, and context limits; actual speed and heat depend on hardware and model.
- Empty local retrieval returns a fixed no-match message. This does not verify factual support for non-empty answers.

## Running the Harness

```bash
# Requires the configured Ollama embedding model; the real LLM check is skipped when unavailable
python eval_harness.py

# Uses stable hash embeddings and skips the real LLM check
python eval_harness.py --mock

# Run one suite and write results to a chosen directory
python eval_harness.py --suite retrieval --out ./eval-output
```

The mock suite still exercises validation paths that perform DNS resolution. The project does not currently commit a dependency lockfile, so package versions can vary between environments.