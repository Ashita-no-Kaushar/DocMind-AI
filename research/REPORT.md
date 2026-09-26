# Agentic map retrieval: ablation and resolution study

Generated 2026-09-25T18:16:52Z by `python -m research.map_ablation`.

## What this measures

Each variant answers the same 31 questions over the same 48 chunks from 8 documents. The map variants are the production `MapRetrievalAgent`; only the agent configuration and the map resolution change between rows. Every variant is capped at the same top-k of 8 results, so the token comparison is like-for-like.

The comparison follows the standard progression from naive retrieval to agentic map navigation:

1. **Naive RAG** — dense top-k retrieval, no agent, no map.
2. **Naive lexical RAG** — BM25 top-k, no agent, no map.
3. **Agentic text RAG** — the same observe/plan/act loop, but over flat chunk cards with no document grouping, hierarchy, or adjacency to navigate.
4. **Agentic map RAG** — the full map agent at low, default, and high resolution, plus component ablations and an oracle upper bound.

Two token costs are reported separately and never merged:

- **Evidence tokens** are what actually enters the model prompt.
- **Map index tokens** are the section cards the agent must score to navigate. For the deterministic planner this is local index work, not prompt tokens. It becomes prompt cost only in the optional LLM-planner mode, which is not run here.

## Ablation results

| Variant | Hit@1 | Hit@k | Precision | Recall | MRR | No-match rejection | Hard-negative FP | Mean evidence tokens | Map index tokens | Total context tokens | Delta % vs naive | Mean steps | Mean nodes scored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Naive vector RAG | 0.84 | 1.00 | 0.14 | 1.00 | 0.91 | 0.00 | 1.00 | 222.0 | 0 | 222.0 | 101.2 | 0.00 | 48.0 |
| Naive lexical RAG (BM25 top-k) | 1.00 | 1.00 | 0.29 | 1.00 | 1.00 | 0.67 | 1.00 | 157.1 | 0 | 157.1 | 0.0 | 0.00 | 48.0 |
| Agentic text RAG (no map) | 1.00 | 1.00 | 0.35 | 1.00 | 1.00 | 0.67 | 1.00 | 104.4 | 0 | 104.4 | -24.4 | 1.87 | 2.8 |
| Map agent (low resolution, 4 chunks/section) | 0.96 | 0.96 | 0.33 | 0.96 | 0.96 | 0.67 | 1.00 | 121.0 | 922 | 1043.0 | -18.9 | 1.23 | 18.6 |
| Map agent (high resolution, 1 chunk/section) | 1.00 | 1.00 | 0.43 | 1.00 | 1.00 | 0.67 | 1.00 | 86.0 | 2484 | 2570.0 | -38.0 | 1.13 | 10.0 |
| Map agent (production default, 6 chunks/section) | 0.92 | 0.92 | 0.29 | 0.92 | 0.92 | 0.67 | 1.00 | 125.7 | 467 | 592.7 | -20.7 | 1.26 | 23.4 |
| Map agent (adaptive, 3 sections/document) | 1.00 | 1.00 | 0.35 | 1.00 | 1.00 | 0.67 | 1.00 | 120.5 | 1358 | 1478.5 | -20.5 | 1.16 | 14.3 |
| Map agent (with candidate refinement) | 0.96 | 0.96 | 0.30 | 0.96 | 0.96 | 0.67 | 1.00 | 132.0 | 467 | 599.0 | -17.2 | 1.16 | 17.4 |
| Map agent (no neighbours) | 0.92 | 0.92 | 0.29 | 0.92 | 0.92 | 0.67 | 1.00 | 125.7 | 467 | 592.7 | -20.7 | 1.26 | 23.4 |
| Map agent (no reflection) | 0.84 | 0.84 | 0.24 | 0.84 | 0.84 | 0.83 | 0.50 | 120.3 | 467 | 587.3 | -28.7 | 1.00 | 12.0 |
| Map agent (single section) | 0.80 | 0.80 | 0.26 | 0.80 | 0.80 | 0.83 | 0.50 | 107.0 | 467 | 574.0 | -35.0 | 1.00 | 6.0 |
| Map agent (oracle planner) | 0.92 | 0.92 | 0.29 | 0.92 | 0.92 | 0.67 | 1.00 | 125.7 | 467 | 592.7 | -20.7 | 1.26 | 23.4 |

Latency is machine-dependent and is recorded in `results/ablation.csv` and `results/ablation.json` rather than in this table, so the report stays byte-reproducible.

## Reading the numbers

- The full map agent used fewer prompt-context tokens than global top-k on answerable questions (-20.7% mean change).
- Hit@1 moved from 1.0 (global top-k) to 0.92 (map agent).
- Reflection fired on 3 answerable questions. The no-reflection ablation scores differently on answerable questions (hit@k 0.84 against 0.92), so the second step is worth a measurable amount.
- The full map agent missed 2 answerable questions (q06, q21).
- `nodes scored` is lexical work only. A hybrid retriever still issues the same embedding query per question, so this is not a total compute saving.
- **Negative result worth reporting:** the structural map did not beat the flat no-map agent on this corpus. The no-map agent reached hit@k 1.00 at 104.4 evidence tokens; the map agent reached 0.92 at 125.7. Most of the token saving here comes from the agentic observe/plan/act loop itself, not from the document structure. A larger corpus with more competing sections is needed before claiming the map adds accuracy.
- Against naive dense retrieval, the map agent moved hit@k from 1.00 to 0.92 and mean evidence tokens from 222.0 to 125.7.

## Map resolution study

Resolution is the number of chunks merged into one map section. More chunks per section means fewer, larger sections: a cheaper index to score but a coarser place to aim. Fewer chunks per section means more, smaller sections: a finer map that costs more to read.

| Resolution | Chunks/section | Sections | Map index tokens | Hit@1 | Hit@k | Precision | Recall | Mean evidence tokens | Total context tokens | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Highest (1 chunk/section) | 1 | 48 | 2484 | 1.00 | 1.00 | 0.43 | 1.00 | 86.0 | 2570.0 | 1.13 |
| High (2 chunks/section) | 2 | 24 | 1358 | 1.00 | 1.00 | 0.35 | 1.00 | 120.5 | 1478.5 | 1.16 |
| Medium-high (3 chunks/section) | 3 | 16 | 927 | 0.92 | 0.92 | 0.31 | 0.92 | 117.5 | 1044.5 | 1.19 |
| Medium (4 chunks/section) | 4 | 16 | 922 | 0.96 | 0.96 | 0.33 | 0.96 | 121.0 | 1043.0 | 1.23 |
| Medium-low (5 chunks/section) | 5 | 16 | 877 | 0.96 | 0.96 | 0.32 | 0.96 | 122.0 | 999.0 | 1.23 |
| Production default (6 chunks/section) | 6 | 8 | 467 | 0.92 | 0.92 | 0.29 | 0.92 | 125.7 | 592.7 | 1.26 |
| Adaptive (3 sections/document) | 6 | 24 | 1358 | 1.00 | 1.00 | 0.35 | 1.00 | 120.5 | 1478.5 | 1.16 |

- Best accuracy at the lowest context cost: Highest (1 chunk/section) (hit@k 1.00, 86.0 evidence tokens).
- Cheapest map to read: Production default (6 chunks/section) (467 map index tokens, hit@k 0.92).
- Going from 6 to 1 chunk per section cuts evidence tokens from 125.7 to 86.0 while raising the map index from 467 to 2484 tokens. The finer map lets the agent aim at a smaller target and so returns less surrounding context.
- Map index tokens are only prompt cost when the optional LLM planner is used. With the deterministic planner they are local scoring work, so the resolution decision trades context size against local compute, not against billed prompt tokens.

Latency is machine-dependent and stays in `results/resolution.csv` and `results/ablation.json` so this report remains byte-reproducible.

See `figures/resolution_tradeoff.svg` for the same data plotted as accuracy against cost.

## Cost model: what the map actually costs

The map is only free if the planner is local. Two models are reported, and they lead to opposite conclusions:

| Variant | Deterministic planner (prompt tokens) | LLM planner upper bound (prompt tokens) | Map index / naive evidence |
| --- | --- | --- | --- |
| Naive vector RAG | 222.0 | 222.0 | 0.00 |
| Naive lexical RAG (BM25 top-k) | 157.1 | 157.1 | 0.00 |
| Agentic text RAG (no map) | 104.4 | 104.4 | 0.00 |
| Map agent (low resolution, 4 chunks/section) | 121.0 | 1043.0 | 5.87 |
| Map agent (high resolution, 1 chunk/section) | 86.0 | 2570.0 | 15.81 |
| Map agent (production default, 6 chunks/section) | 125.7 | 592.7 | 2.97 |
| Map agent (adaptive, 3 sections/document) | 120.5 | 1478.5 | 8.65 |
| Map agent (with candidate refinement) | 132.0 | 599.0 | 2.97 |
| Map agent (no neighbours) | 125.7 | 592.7 | 2.97 |
| Map agent (no reflection) | 120.3 | 587.3 | 2.97 |
| Map agent (single section) | 107.0 | 574.0 | 2.97 |
| Map agent (oracle planner) | 125.7 | 592.7 | 2.97 |

- Naive lexical top-k costs 157.1 prompt tokens per question. That figure is capped by top-k and does not grow with the corpus.
- Map agent (low resolution, 4 chunks/section): the map index is 5.87x the whole naive context, so charging the map to the prompt costs 886.0 extra tokens per question. Under the LLM-planner model this variant is a net loss.
- Map agent (high resolution, 1 chunk/section): the map index is 15.81x the whole naive context, so charging the map to the prompt costs 2412.9 extra tokens per question. Under the LLM-planner model this variant is a net loss.
- Map agent (production default, 6 chunks/section): the map index is 2.97x the whole naive context, so charging the map to the prompt costs 435.6 extra tokens per question. Under the LLM-planner model this variant is a net loss.
- Map agent (adaptive, 3 sections/document): the map index is 8.65x the whole naive context, so charging the map to the prompt costs 1321.4 extra tokens per question. Under the LLM-planner model this variant is a net loss.
- Map agent (with candidate refinement): the map index is 2.97x the whole naive context, so charging the map to the prompt costs 441.9 extra tokens per question. Under the LLM-planner model this variant is a net loss.
- Map agent (no neighbours): the map index is 2.97x the whole naive context, so charging the map to the prompt costs 435.6 extra tokens per question. Under the LLM-planner model this variant is a net loss.
- Map agent (no reflection): the map index is 2.97x the whole naive context, so charging the map to the prompt costs 430.2 extra tokens per question. Under the LLM-planner model this variant is a net loss.
- Map agent (single section): the map index is 2.97x the whole naive context, so charging the map to the prompt costs 417.0 extra tokens per question. Under the LLM-planner model this variant is a net loss.
- **Break-even condition.** The map only pays for itself under the LLM-planner model when `map_index_tokens < naive_evidence_tokens - map_evidence_tokens`. Because naive top-k evidence is capped by top-k while the map index grows with the number of sections, this condition is harder to satisfy as the corpus grows, not easier.
- **Therefore the honest economic claim is not a token saving.** The defensible claim is that the deterministic planner performs structure-guided routing with zero extra model calls: the map is scored locally, so the only prompt cost is the routed evidence, which is at most the naive cost. That framing is what the application default implements.
- A reviewer should read the LLM-planner column as the price of the alternative design, not as a cost the default configuration pays.

## Scaling study: does the map help as the corpus grows?

The question set is held fixed while distractor documents that reuse the corpus vocabulary are added, so every change in hit rate comes from corpus difficulty rather than easier questions.

| Documents | Chunks | Variant | Hit@1 | Hit@k | Precision | Mean evidence tokens | Map index tokens | Routing engaged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 48 | Naive lexical RAG (BM25 top-k) | 1.00 | 1.00 | 0.29 | 157.1 | 0 | 0.00 |
| 8 | 48 | Agentic text RAG (no map) | 1.00 | 1.00 | 0.35 | 104.4 | 0 | 0.00 |
| 8 | 48 | Map agent (production default) | 0.92 | 0.92 | 0.29 | 125.7 | 467 | 0.92 |
| 8 | 48 | Map agent (adaptive) | 1.00 | 1.00 | 0.35 | 120.5 | 1358 | 1.00 |
| 8 | 48 | Map agent (high resolution) | 1.00 | 1.00 | 0.43 | 86.0 | 2484 | 1.00 |
| 16 | 96 | Naive lexical RAG (BM25 top-k) | 1.00 | 1.00 | 0.21 | 169.5 | 0 | 0.00 |
| 16 | 96 | Agentic text RAG (no map) | 1.00 | 1.00 | 0.29 | 105.0 | 0 | 0.00 |
| 16 | 96 | Map agent (production default) | 0.88 | 0.88 | 0.23 | 138.8 | 962 | 0.96 |
| 16 | 96 | Map agent (adaptive) | 1.00 | 1.00 | 0.32 | 121.0 | 2781 | 1.00 |
| 16 | 96 | Map agent (high resolution) | 1.00 | 1.00 | 0.42 | 83.1 | 4710 | 1.00 |
| 32 | 192 | Naive lexical RAG (BM25 top-k) | 1.00 | 1.00 | 0.21 | 172.0 | 0 | 0.00 |
| 32 | 192 | Agentic text RAG (no map) | 1.00 | 1.00 | 0.29 | 104.8 | 0 | 0.00 |
| 32 | 192 | Map agent (production default) | 0.88 | 0.88 | 0.22 | 141.8 | 1927 | 0.96 |
| 32 | 192 | Map agent (adaptive) | 1.00 | 1.00 | 0.32 | 121.0 | 5656 | 1.00 |
| 32 | 192 | Map agent (high resolution) | 1.00 | 1.00 | 0.42 | 83.1 | 6201 | 1.00 |
| 64 | 384 | Naive lexical RAG (BM25 top-k) | 0.96 | 1.00 | 0.21 | 172.3 | 0 | 0.00 |
| 64 | 384 | Agentic text RAG (no map) | 0.96 | 1.00 | 0.29 | 105.1 | 0 | 0.00 |
| 64 | 384 | Map agent (production default) | 0.84 | 0.88 | 0.22 | 141.8 | 3870 | 0.96 |
| 64 | 384 | Map agent (adaptive) | 0.96 | 1.00 | 0.32 | 121.0 | 7586 | 1.00 |
| 64 | 384 | Map agent (high resolution) | 0.96 | 1.00 | 0.42 | 83.1 | 6201 | 1.00 |

- 8 documents: best hit@k 1.00 shared by naive, flat agent, map agent; the best map variant is Map agent (high resolution).
- 16 documents: best hit@k 1.00 shared by naive, flat agent, map agent; the best map variant is Map agent (high resolution).
- 32 documents: best hit@k 1.00 shared by naive, flat agent, map agent; the best map variant is Map agent (high resolution).
- 64 documents: best hit@k 1.00 shared by naive, flat agent, map agent; the best map variant is Map agent (high resolution).
- **No accuracy crossover was observed.** Naive top-k and the flat agent stay at the top of this question set at every corpus size, so this study does not demonstrate a point where routing finds answers that top-k misses. Proving that needs documents long enough that the answer is one of hundreds of chunks, or genuinely multi-hop questions, neither of which this synthetic corpus contains.
- The map's demonstrated benefit is different and still real: at equal accuracy it returns a much smaller context, and that gap widens as the corpus grows.
- 8 documents: Map agent (high resolution) matches naive accuracy at 86.0 evidence tokens against 157.1, a 45.3% smaller prompt, for 2484 map index tokens of local work.
- 16 documents: Map agent (high resolution) matches naive accuracy at 83.1 evidence tokens against 169.5, a 51.0% smaller prompt, for 4710 map index tokens of local work.
- 32 documents: Map agent (high resolution) matches naive accuracy at 83.1 evidence tokens against 172.0, a 51.7% smaller prompt, for 6201 map index tokens of local work.
- 64 documents: Map agent (high resolution) matches naive accuracy at 83.1 evidence tokens against 172.3, a 51.8% smaller prompt, for 6201 map index tokens of local work.
- Map index tokens grow with the number of sections while naive prompt cost stays flat, because naive retrieval is capped at top-k. This is the break-even condition from the cost-model section, measured rather than assumed.
- The fixed production resolution is the weakest setting and degrades as the corpus grows, while the adaptive setting holds its hit rate. That is the practical recommendation from this study.

## Limitations

- Resolution here means **structural granularity of the layout map**, the number of chunks per map section. This is not pixel or image resolution: no page image, figure, or table region is embedded or indexed. A multimodal page-image resolution study is separate work and no result for it is claimed.
- The retrieval backends are deterministic fixtures: BM25 for the lexical rows and hash embeddings for the dense row. The dense row is a reproducible stand-in for a trained encoder, not a real embedding model, so its absolute numbers are not a claim about any production model.
- The corpus is 8 synthetic policy documents with 25 answerable and 6 no-match questions. These are small numbers and the confidence intervals are wide.
- The oracle planner variant is an upper bound that reads ground truth. It is not an LLM result and must never be reported as one.
- The optional LLM planner was not executed in this run: no endpoint supplied.
- Token counts come from a whitespace tokenizer, not a provider tokenizer.
- Prompt-context tokens are a narrow claim: this study does not show better answers, lower total cost, or lower end-to-end latency.

## Reproduce

```bash
python -m research.map_ablation
python -m research.map_ablation --check
```

Corpus fingerprint `115e88f9c6a8ec58f70dbecddb6295ede51bd5094b36363ad437e40376580495`.
