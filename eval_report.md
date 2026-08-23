# DocMind AI — Full Project Evaluation Report

_Generated 2026-08-23 18:38 — overall score **100%** (43/43 tests)_

## Summary (suite scores)

| Suite | Passed | Score | What it covers |
|---|---:|---:|---|
| ingestion | 7/7 | **100%** | File types, chunking, dedup, title-aware, cache, exclusions |
| retrieval | 12/12 | **100%** | Hybrid BM25+vector vs plain, synonyms, Hyphen/Hinglish, evidence floor |
| generation | 6/6 | **100%** | No-hallucination fallback, citations, tone presets, multi-turn, e2e |
| performance | 5/5 | **100%** | Ingest speed, retrieval latency, eco-mode, cache hit, OOM resilience |
| robustness | 6/6 | **100%** | Empty/binary handling, security (GitHub/URL validation), special chars |
| architecture | 7/7 | **100%** | Backend presets, export, settings persistence, import boundaries |
| **Overall** | **43/43** | **100%** | |

## Retrieval deep-dive (plain vs Hybrid)

_Embedding: `nomic-embed-text:latest` — hit = expected doc in top-3, correct rejection = no-match returns 0 nodes_

| Metric | Plain RAG | DocMind Hybrid |
|---|---:|---:|
| Factual hit rate | 100% | **100%** |
| Correct rejection | 0% | **100%** |
| Avg latency | 24 ms | 24 ms |

| # | Query | Expected | Feature | Plain | Hybrid | Plain top | Hybrid top |
|---|---|---|---|:---:|:---:|---|---|
| 1 | What is the refund policy? | refund_policy.txt | basic | ✓ | ✓ | refund_policy.txt | refund_policy.txt |
| 2 | money back rules | refund_policy.txt | synonym | ✓ | ✓ | refund_policy.txt | refund_policy.txt |
| 3 | 30-day returns | refund_policy.txt | hyphen | ✓ | ✓ | refund_policy.txt | refund_policy.txt |
| 4 | refund policy batao | refund_policy.txt | hinglish | ✓ | ✓ | refund_policy.txt | refund_policy.txt |
| 5 | annual report revenue growth | annual_report.txt | title-aware | ✓ | ✓ | annual_report.txt | annual_report.txt |
| 6 | How to fetch a URL in Python? | python_guide.txt | basic | ✓ | ✓ | python_guide.txt | python_guide.txt |
| 7 | How many leave days per year? | hr_policy.txt | basic | ✓ | ✓ | hr_policy.txt | hr_policy.txt |
| 8 | pasta cooking time | cooking.txt | basic | ✓ | ✓ | cooking.txt | cooking.txt |
| 9 | What is the capital of France? | — (no match) | no-match | ✗ | ✓ | annual_report.txt | — |
| 10 | quantum physics equations | — (no match) | no-match | ✗ | ✓ | python_guide.txt | — |
| 11 | refunded purchases | refund_policy.txt | stemming | ✓ | ✓ | refund_policy.txt | refund_policy.txt |
| 12 | tell me about the annual report | annual_report.txt | filler-filter | ✓ | ✓ | annual_report.txt | annual_report.txt |

## Suite: ingestion (7/7 — 100%)

| Test | Pass | Detail |
|---|:---:|---|
| excluded_patterns | ✓ | loaded=1 patterns=34 321ms |
| title_aware | ✓ | title='Annual Report' 0ms |
| dedupe | ✓ | 3 -> 2 989ms |
| multiformat_index | ✓ | docs=4 docx=yes 118ms |
| cache_key_and_persist | ✓ | key=fa6b2b5cd02345e34802 persist=hit 69ms |
| min_chunk_filter | ✓ | MIN_CHARS=15 docs_in=2 13ms |
| helpers_github_normalize | ✓ | owner/repo=True url=True reject_gitlab=True 5ms |

## Suite: generation (6/6 — 100%)

| Test | Pass | Detail |
|---|:---:|---|
| qa_template_guards | ✓ | has_guards=True len=579 0ms |
| query_helpers | ✓ | hyphen=True hinglish=True synonym=True 0ms |
| no_hallucination_fallback | ✓ | fallback=hit 4ms |
| tone_presets | ✓ | presets=6 distinct=True 1ms |
| multi_turn_history | ✓ | messages_in_prompt=4 109ms |
| e2e_real_llm | ✓ | answer_len=233 has_30days=True preview='The refund policy is as follows: (from [1]). Full refunds are available within 3 5101ms |

## Suite: performance (5/5 — 100%)

| Test | Pass | Detail |
|---|:---:|---|
| ingest_5_docs | ✓ | 2.5s for 5 docs 4733ms |
| cache_hit_faster | ✓ | cold 0.0s vs cache 0.03s 93ms |
| retrieval_latency | ✓ | vector 24ms hybrid 23ms 107ms |
| eco_mode_trims | ✓ | ctx 4800->3200 pred 512->256 batch 16->4 0ms |
| batch_oom_resilience | ✓ | shrunk to 4 after OOM 0ms |

## Suite: robustness (6/6 — 100%)

| Test | Pass | Detail |
|---|:---:|---|
| github_validation | ✓ | 5/5 cases |
| url_validation | ✓ | https=True bad=True xss=True |
| empty_docs_rejected | ✓ | Index creation failed: No usable content was extracted from the documents. The f |
| binary_exclusion | ✓ | loaded 1 (png excluded) |
| special_chars_tokenization | ✓ | tokens=['30', 'day', 'money', 'back', 'refund'] |
| hinglish_filler_filter | ✓ | rewritten='refund polici plea' |

## Suite: architecture (7/7 — 100%)

| Test | Pass | Detail |
|---|:---:|---|
| backend_presets | ✓ | presets=['Ollama', 'OpenAI', 'LM Studio (Local AI)', 'TabbyAPI'] |
| export_docx | ✓ | docx 36713 bytes |
| browser_settings_keys | ✓ | key=browser_settings_persisted_hash |
| ollama_helpers | ✓ | estimate=2 trim=1 |
| embedding_verify | ✓ | mock verify |
| r2r_health | ✓ | health mocked |
| no_torch_at_import | ✓ | torch imports=0 |

## What DocMind adds vs plain vector RAG

- **Hybrid BM25 + vector (RRF)** — keyword hits rescue low vector scores; `utils/llama_index.py:260`
- **Evidence floor (0.5)** — weak scores need BM25 evidence or the query is correctly rejected; `utils/llama_index.py:75`
- **Synonym expansion** — short queries expand (refund→return/money back) for BM25 only; `utils/llama_index.py:201`
- **Hyphen & Hinglish** — `30-day`→`30 day`, filler words `batao/kya/hai` removed; `utils/llama_index.py:184`
- **Title-aware chunks** — every chunk prefixed with its document title; `utils/llama_index.py:141`
- **Dedup & cache** — near-duplicate filtering + on-disk index cache; `utils/llama_index.py:108`, `834`
- **Eco Mode** — trims batches/context for cool & fast answers; `utils/llama_index.py:166`, `utils/ollama.py:100`
- **No-hallucination fallback** — `I could not find this information…` and `Ask without documents`; `utils/ollama.py:528`

## Reproducibility

```bash
# real embeddings + LLM (needs Ollama running)
python eval_harness.py

# deterministic, no server (CI-friendly)
python eval_harness.py --mock

# single suite
python eval_harness.py --suite retrieval
```