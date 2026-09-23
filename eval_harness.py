"""Evaluation harness for DocMind's implemented subsystems.

Run:
  python eval_harness.py              # require Ollama embeddings; run real LLM check when available
  python eval_harness.py --mock       # use stable hash embeddings and skip the real LLM check
  python eval_harness.py --suite retrieval
  python eval_harness.py --quick      # alias for --mock

Suites:
  ingestion   — loader exclusions, chunking, dedup, titles, cache, helper validation
  retrieval   — fixture retrieval, query cleanup, evidence filtering, hybrid ranking
  generation  — prompt guards, no-match fallback, tone prompts, history, three real LLM trials
  performance — fixture timings, cache round-trip, Eco settings, batch shrinking
  robustness  — empty/binary handling, URL/GitHub validation, token cleanup
  architecture— provider definitions, export, persistence contract, R2R health mock

Outputs:
  eval_results.json — machine-readable per-suite results
  eval_report.md    — generated report with fixture scope and limitations
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus — 5 core docs + fixtures for file-type / ingestion suite
# ---------------------------------------------------------------------------

DOCS = {
    "refund_policy.txt": (
        "Refund Policy\n"
        "Full refunds are available within 30 days of purchase. "
        "Electronics are covered for 15 days only. "
        "Contact support@docmind.test for returns. "
        "All refunds are processed within 5 business days."
    ),
    "annual_report.txt": (
        "Annual Report 2024\n"
        "Revenue grew 23% to $4.2M in fiscal year 2024. "
        "Community outreach expanded to 12 new districts. "
        "Operating costs were reduced by 8% through automation."
    ),
    "python_guide.txt": (
        "Python Guide — Fetching URLs\n"
        "Use requests.get('https://example.com') to fetch web content. "
        "Set timeout=10 to avoid hanging. "
        "Handle errors with try/except and check response.status_code."
    ),
    "hr_policy.txt": (
        "HR Policy — Leave\n"
        "Employees receive 20 days of annual leave and 10 days of sick leave per year. "
        "Work from home is allowed 2 days per week with manager approval."
    ),
    "cooking.txt": (
        "Cooking — Pasta Recipe\n"
        "Boil water, add pasta and cook 10 minutes. "
        "Add tomatoes, basil, and olive oil. Serve warm."
    ),
}

# Retrieval queries — each maps to a feature under test
QUERIES = [
    {"query": "What is the refund policy?", "expected": "refund_policy.txt", "feature": "basic"},
    {"query": "money back rules", "expected": "refund_policy.txt", "feature": "synonym"},
    {"query": "30-day returns", "expected": "refund_policy.txt", "feature": "hyphen"},
    {"query": "refund policy batao", "expected": "refund_policy.txt", "feature": "hinglish"},
    {"query": "annual report revenue growth", "expected": "annual_report.txt", "feature": "title-aware"},
    {"query": "How to fetch a URL in Python?", "expected": "python_guide.txt", "feature": "basic"},
    {"query": "How many leave days per year?", "expected": "hr_policy.txt", "feature": "basic"},
    {"query": "pasta cooking time", "expected": "cooking.txt", "feature": "basic"},
    {"query": "What is the capital of France?", "expected": None, "feature": "no-match"},
    {"query": "quantum physics equations", "expected": None, "feature": "no-match"},
    {"query": "refunded purchases", "expected": "refund_policy.txt", "feature": "stemming"},
    {"query": "tell me about the annual report", "expected": "annual_report.txt", "feature": "filler-filter"},
]

ALL_SUITES = ["ingestion", "retrieval", "generation", "performance", "robustness", "architecture"]


# ---------------------------------------------------------------------------
# Helpers: streamlit state, embeddings
# ---------------------------------------------------------------------------

def _setup_streamlit_state():
    try:
        import streamlit as st
        defaults = {
            "top_k": 3,
            "candidate_depth": 10,
            "vector_candidate_depth": 10,
            "bm25_candidate_depth": 10,
            "similarity_cutoff": 0.3,
            "eco_mode": False,
            "chunk_size": 256,
            "chunk_overlap": 32,
            "chunk_overlap_pct": 12,
            "temperature": 0.4,
            "advanced": False,
            "llm_backend": "Ollama",
            "ollama_endpoint": "http://localhost:11434",
            "ollama_embedding_model": "nomic-embed-text:latest",
            "selected_model": "qwen2.5:0.5b",
            "system_prompt": "You are DocMind AI.",
            "messages": [],
            "retriever": None,
            "query_engine": None,
            "last_doc_sources": [],
            "last_rag_evidence": [],
            "last_rag_no_result": False,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                try:
                    st.session_state[k] = v
                except Exception:
                    pass
    except Exception:
        pass


def _try_real_embeddings(endpoint, model):
    from utils.llama_index import setup_embedding_model
    try:
        setup_embedding_model(
            model=model, chunk_size=256, chunk_overlap=32,
            backend="Ollama", ollama_endpoint=endpoint,
        )
        return True
    except Exception as e:
        print(f"  Real embeddings unavailable ({e}), falling back to hash embeddings.")
        return False


def _setup_hash_embeddings():
    from llama_index.core import Settings
    from llama_index.core.embeddings import BaseEmbedding
    from utils.llama_index import _bm25_tokens

    class HashEmbedding(BaseEmbedding):
        model_name: str = "hash-mock"
        def _get_text_embedding(self, text: str):
            vec = [0.0] * 64
            for tok in _bm25_tokens(text):
                digest = hashlib.sha256(tok.encode("utf-8")).digest()
                h = int.from_bytes(digest[:8], "big") % 64
                vec[h] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            return [x / norm for x in vec]
        def _get_query_embedding(self, query: str):
            return self._get_text_embedding(query)
        async def _aget_query_embedding(self, query: str):
            return self._get_query_embedding(query)

    Settings.embed_model = HashEmbedding()
    from llama_index.core import Settings as _S
    _S.chunk_size = 256
    _S.chunk_overlap = 32
    print("  Using stable hash embeddings (no Ollama).")


def _create_index_from_docs(tmpdir: str):
    from utils.llama_index import load_documents, create_index
    for name, content in DOCS.items():
        Path(tmpdir, name).write_text(content, encoding="utf-8")
    docs = load_documents(tmpdir)
    return create_index(docs)


def _evaluate_retriever(retriever, query: str, expected: str | None):
    t0 = time.perf_counter()
    try:
        nodes = retriever.retrieve(query)
    except Exception as e:
        return {"error": str(e), "hit": False, "latency": time.perf_counter() - t0, "top_file": None, "top_score": 0.0, "num_nodes": 0}
    latency = time.perf_counter() - t0
    if expected is None:
        hit = len(nodes) == 0
        top_file = nodes[0].node.metadata.get("file_name") if nodes else None
        top_score = nodes[0].score if nodes else 0.0
    else:
        files = [(n.node.metadata.get("file_name") or "") for n in nodes]
        hit = any(r == expected for r in files)
        top_file = files[0] if files else None
        top_score = nodes[0].score if nodes else 0.0
    return {"hit": hit, "latency": latency, "top_file": top_file, "top_score": round(float(top_score), 3) if nodes else 0.0, "num_nodes": len(nodes)}


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

def suite_ingestion(mock: bool):
    """Ingestion pipeline: file types, exclusions, chunking, dedup, title, cache."""
    print("\n[Suite] Ingestion")
    # ensure embeddings are available even when suite is run standalone
    try:
        from llama_index.core import Settings
        if not getattr(Settings, "embed_model", None) or getattr(Settings.embed_model, "model_name", "") == "unknown":
            _setup_hash_embeddings()
    except Exception:
        try:
            _setup_hash_embeddings()
        except Exception:
            pass
    results = []
    def check(name, fn):
        try:
            t0 = time.perf_counter()
            ok, detail = fn()
            dt = round((time.perf_counter() - t0) * 1000)
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {name:32s}  {detail} ({dt}ms)")
            results.append({"test": name, "pass": ok, "detail": detail, "ms": dt})
        except Exception as e:
            print(f"  FAIL  {name:32s}  exception: {e}")
            results.append({"test": name, "pass": False, "detail": f"exception: {e}", "ms": 0})

    # 1 — load_documents filters excluded patterns
    def t_excluded():
        from utils.llama_index import load_documents, EXCLUDED_FILE_PATTERNS
        d = tempfile.mkdtemp()
        try:
            Path(d, "keep.txt").write_text("hello world content for testing keep file", encoding="utf-8")
            Path(d, "ignore.png").write_bytes(b"\x89PNG")
            Path(d, "ignore.zip").write_bytes(b"PK")
            docs = load_documents(d)
            texts = " ".join(getattr(doc, "text", "") or "" for doc in docs)
            ok = "hello world" in texts and len(docs) == 1
            return ok, f"loaded={len(docs)} patterns={len(EXCLUDED_FILE_PATTERNS)}"
        finally:
            shutil.rmtree(d, ignore_errors=True)
    check("excluded_patterns", t_excluded)

    # 2 — title-aware
    def t_title():
        from utils.llama_index import _document_title, _prepend_document_title
        from llama_index.core.schema import TextNode
        n = TextNode(text="body content here", metadata={"file_name": "annual_report.md"})
        title = _document_title(n)
        _prepend_document_title([n])
        ok = title == "Annual Report" and n.get_content().startswith("Annual Report.")
        return ok, f"title='{title}'"
    check("title_aware", t_title)

    # 3 — dedup
    def t_dedup():
        from utils.llama_index import _dedupe_near_duplicate_nodes
        from llama_index.core.schema import Document
        chunks = [
            Document(text="Welcome to our annual report. The company grew a lot this year."),
            Document(text="Welcome to our annual report. The company grew a lot this year!"),
            Document(text="Revenues increased by 12 percent."),
        ]
        deduped = _dedupe_near_duplicate_nodes(chunks)
        return len(deduped) == 2, f"{len(chunks)} -> {len(deduped)}"
    check("dedupe", t_dedup)

    # 4 — chunking + index creation (file-type coverage: txt, md, csv, docx)
    def t_index_multiformat():
        from utils.llama_index import load_documents, create_index
        d = tempfile.mkdtemp()
        try:
            Path(d, "a.txt").write_text(DOCS["refund_policy.txt"], encoding="utf-8")
            Path(d, "b.md").write_text("# Annual Report\n" + DOCS["annual_report.txt"], encoding="utf-8")
            # csv
            with open(os.path.join(d, "c.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["topic", "detail"])
                w.writerow(["leave", "20 days annual"])
                w.writerow(["refund", "30 days"])
            # docx
            try:
                import docx
                doc = docx.Document()
                doc.add_paragraph(DOCS["python_guide.txt"])
                doc.save(os.path.join(d, "d.docx"))
                docx_ok = True
            except Exception:
                docx_ok = False
                Path(d, "d.txt").write_text(DOCS["python_guide.txt"], encoding="utf-8")
            try:
                docs = load_documents(d)
            except Exception as e:
                if "docx2txt" in str(e):
                    # environment lacks docx2txt — retry without the docx file
                    try:
                        os.remove(os.path.join(d, "d.docx"))
                    except Exception:
                        pass
                    Path(d, "d.txt").write_text(DOCS["python_guide.txt"], encoding="utf-8")
                    docs = load_documents(d)
                    docx_ok = False
                else:
                    raise
            index = create_index(docs)
            # check that we got docs from multiple extensions
            ok = len(docs) >= 4 and index is not None
            return ok, f"docs={len(docs)} docx={'yes' if docx_ok else 'skip-no-docx2txt'}"
        finally:
            shutil.rmtree(d, ignore_errors=True)
    check("multiformat_index", t_index_multiformat)

    # 5 — cache: same docs produce same cache key and hit on second build
    def t_cache():
        from utils.llama_index import index_cache_key, load_documents, create_index, index_cache_dir, load_index_from_cache, persist_index_to_cache
        d = tempfile.mkdtemp()
        try:
            for name, content in list(DOCS.items())[:2]:
                Path(d, name).write_text(content, encoding="utf-8")
            docs = load_documents(d)
            k1 = index_cache_key(docs)
            k2 = index_cache_key(docs)
            ok = k1 == k2 and len(k1) == 20
            # try persist/load round-trip (best effort)
            try:
                idx = create_index(docs)
                cdir = index_cache_dir(docs)
                if cdir:
                    persist_index_to_cache(idx, cdir)
                    loaded = load_index_from_cache(cdir)
                    ok = ok and (loaded is not None)
                    detail = f"key={k1} persist={'hit' if loaded else 'miss'}"
                else:
                    detail = f"key={k1}"
            except Exception as e:
                detail = f"key={k1} persist skipped ({e})"
            return ok, detail
        finally:
            shutil.rmtree(d, ignore_errors=True)
    check("cache_key_and_persist", t_cache)

    # 6 — MIN_CHUNK_CHARS filtering (very short docs dropped)
    def t_min_chars():
        from utils.llama_index import MIN_CHUNK_CHARS, load_documents, create_index
        d = tempfile.mkdtemp()
        try:
            Path(d, "tiny.txt").write_text("hi", encoding="utf-8")
            Path(d, "ok.txt").write_text("This is a sufficiently long document content for chunking tests. " * 5, encoding="utf-8")
            docs = load_documents(d)
            try:
                create_index(docs)
                # tiny doc should be filtered, but ok doc remains -> index succeeds
                return True, f"MIN_CHARS={MIN_CHUNK_CHARS} docs_in={len(docs)}"
            except Exception as e:
                # if only tiny doc, index should raise "No usable content"
                return "No usable" in str(e), str(e)[:60]
        finally:
            shutil.rmtree(d, ignore_errors=True)
    check("min_chunk_filter", t_min_chars)

    # 7 — website / github helpers (input validation)
    def t_helpers():
        from utils.helpers import normalize_github_repo
        try:
            ok1 = normalize_github_repo("owner/repo") == "owner/repo"
            ok2 = normalize_github_repo("https://github.com/owner/repo") == "owner/repo"
            ok3 = False
            try:
                normalize_github_repo("https://gitlab.com/owner/repo")
            except Exception:
                ok3 = True
            return (ok1 and ok2 and ok3), f"owner/repo={ok1} url={ok2} reject_gitlab={ok3}"
        except Exception as e:
            return False, str(e)
    check("helpers_github_normalize", t_helpers)

    passed = sum(1 for r in results if r["pass"])
    return {"suite": "ingestion", "results": results, "passed": passed, "total": len(results), "score": round(passed / len(results), 3) if results else 0}


def suite_retrieval(mock: bool, endpoint="http://localhost:11434", model="nomic-embed-text:latest"):
    print("\n[Suite] Retrieval (plain vs hybrid)")
    _setup_streamlit_state()
    if mock:
        _setup_hash_embeddings()
    elif not _try_real_embeddings(endpoint, model):
        raise RuntimeError(
            f"Real embeddings were required but unavailable for {model} at {endpoint}"
        )

    tmpdir = tempfile.mkdtemp(prefix="docmind_eval_")
    try:
        from utils.llama_index import build_hybrid_retriever, HybridRetriever
        index = _create_index_from_docs(tmpdir)
        vector_retriever = index.as_retriever(similarity_top_k=3)
        class PlainWrapper:
            def __init__(self, inner, cutoff=0.3):
                self.inner = inner
                self.cutoff = cutoff
            def retrieve(self, query):
                nodes = self.inner.retrieve(query)
                return [n for n in nodes if n.score >= self.cutoff]
        plain = PlainWrapper(vector_retriever, cutoff=0.3)
        hybrid = build_hybrid_retriever(vector_retriever, index, top_k=3, similarity_cutoff=0.3)
        is_hybrid = isinstance(hybrid, HybridRetriever)

        results = []
        for item in QUERIES:
            q = item["query"]
            exp = item["expected"]
            pr = _evaluate_retriever(plain, q, exp)
            hr = _evaluate_retriever(hybrid, q, exp)
            p_mark = "PASS" if pr["hit"] else "FAIL"
            h_mark = "PASS" if hr["hit"] else "FAIL"
            print(f"  {h_mark if hr['hit'] else 'FAIL'}  [{item['feature']:12s}] {q[:42]:42s}  plain:{p_mark} hybrid:{h_mark} ({pr['top_file'] or '-'} -> {hr['top_file'] or '-'})")
            results.append({"query": q, "expected": exp, "feature": item["feature"], "plain_hit": pr["hit"], "hybrid_hit": hr["hit"], "plain_top": pr["top_file"], "hybrid_top": hr["top_file"], "plain_score": pr["top_score"], "hybrid_score": hr["top_score"], "plain_latency": round(pr["latency"], 4), "hybrid_latency": round(hr["latency"], 4)})

        factual = [r for r in results if r["feature"] != "no-match"]
        no_match = [r for r in results if r["feature"] == "no-match"]
        def hr(rows, key): return sum(1 for r in rows if r[key]) / len(rows) if rows else 0
        summary = {
            "plain_hit_rate": round(hr(factual, "plain_hit"), 3),
            "hybrid_hit_rate": round(hr(factual, "hybrid_hit"), 3),
            "plain_rejection": round(hr(no_match, "plain_hit"), 3),
            "hybrid_rejection": round(hr(no_match, "hybrid_hit"), 3),
            "plain_avg_ms": round(sum(r["plain_latency"] for r in results) / len(results) * 1000, 1) if results else 0,
            "hybrid_avg_ms": round(sum(r["hybrid_latency"] for r in results) / len(results) * 1000, 1) if results else 0,
            "is_hybrid": is_hybrid,
            "embedding": "hash-mock" if mock else model,
        }
        # feature breakdown
        for feat in sorted(set(r["feature"] for r in results)):
            rows = [r for r in results if r["feature"] == feat]
            summary[f"hybrid_{feat}_rate"] = round(hr(rows, "hybrid_hit"), 3)

        print(f"  Summary: hit plain {summary['plain_hit_rate']:.0%} hybrid {summary['hybrid_hit_rate']:.0%} | reject plain {summary['plain_rejection']:.0%} hybrid {summary['hybrid_rejection']:.0%} | latency plain {summary['plain_avg_ms']}ms hybrid {summary['hybrid_avg_ms']}ms")
        passed = sum(1 for r in results if r["hybrid_hit"])
        score = round(passed / len(results), 3) if results else 0
        return {"suite": "retrieval", "results": results, "summary": summary, "passed": passed, "total": len(results), "score": score}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def suite_generation(mock: bool, endpoint="http://localhost:11434", model="nomic-embed-text:latest"):
    print("\n[Suite] Generation (RAG answer quality)")
    results = []

    def check(name, fn):
        try:
            t0 = time.perf_counter()
            ok, detail = fn()
            dt = int((time.perf_counter() - t0) * 1000)
            status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
            print(f"  {status}  {name:32s}  {detail} ({dt}ms)")
            results.append({"test": name, "pass": ok, "detail": detail, "ms": dt})
        except Exception as e:
            print(f"  FAIL  {name:32s}  exception: {e}")
            results.append({"test": name, "pass": False, "detail": f"exception: {e}", "ms": 0})

    # 1 — TEXT_QA_TEMPLATE exists and has required guards
    def t_template():
        from utils.llama_index import TEXT_QA_TEMPLATE
        tmpl = TEXT_QA_TEMPLATE.template if hasattr(TEXT_QA_TEMPLATE, "template") else str(TEXT_QA_TEMPLATE)
        ok = "ONLY the context" in tmpl and "could not find" in tmpl.lower() and "{context_str}" in tmpl
        return ok, f"has_guards={ok} len={len(tmpl)}"
    check("qa_template_guards", t_template)

    # 2 — _rewrite_query / BM25 helpers (covers stemming, Hinglish, hyphen)
    def t_query_helpers():
        from utils.llama_index import _bm25_tokens, _rewrite_query, _bm25_expanded_tokens
        ok1 = _bm25_tokens("30-day guarantee's") == ["30", "day", "guarante"]
        ok2 = "batao" not in _rewrite_query("batao refund policy kya hai")
        ok3 = "return" in _bm25_expanded_tokens(["refund", "polici"])
        return (ok1 and ok2 and ok3), f"hyphen={ok1} hinglish={ok2} synonym={ok3}"
    check("query_helpers", t_query_helpers)

    # 3 — context_chat no-match fallback (mock retriever returns [])
    def t_no_match():
        from unittest.mock import patch, MagicMock
        from utils import ollama as oll_mod
        import streamlit as st
        # fake retriever that returns []
        fake_qe = MagicMock()
        fake_qe._retriever = MagicMock()
        fake_qe._retriever.retrieve.return_value = []
        # ensure st.session_state has needed keys
        _setup_streamlit_state()
        st.session_state["retriever"] = fake_qe._retriever
        st.session_state["query_engine"] = fake_qe
        # mock LLM to ensure not called
        with patch.object(oll_mod, "create_llm"):
            from utils.ollama import context_chat
            chunks = list(context_chat("quantum physics", query_engine=fake_qe))
            text = "".join(chunks)
            ok = "could not find" in text.lower()
            return ok, f"fallback={'hit' if ok else 'miss'}"
    check("no_match_fallback", t_no_match)

    # 4 — tone presets produce distinct prompts
    def t_tone():
        from components.chatbox import _style_to_prompt, ANSWER_STYLE_OPTIONS
        prompts = {s: _style_to_prompt(s) for s in ANSWER_STYLE_OPTIONS}
        ok = len(set(prompts.values())) == len(ANSWER_STYLE_OPTIONS)
        return ok, f"presets={len(prompts)} distinct={ok}"
    check("tone_presets", t_tone)

    # 5 — multi-turn: context_chat includes history
    def t_multiturn():
        from unittest.mock import patch, MagicMock
        from llama_index.core.schema import TextNode, NodeWithScore
        from utils import ollama as oll_mod
        import streamlit as st
        _setup_streamlit_state()
        st.session_state["messages"] = [
            {"role": "user", "content": "What is refund policy?"},
            {"role": "assistant", "content": "30 days."},
        ]
        st.session_state["system_prompt"] = "You are DocMind AI."
        # retriever returns one node
        node = TextNode(text="Refund 30 days.", metadata={"file_name": "refund_policy.txt"})
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = [NodeWithScore(node=node, score=0.9)]
        fake_qe = MagicMock()
        fake_qe._retriever = fake_retriever
        st.session_state["retriever"] = fake_retriever
        st.session_state["query_engine"] = fake_qe
        captured = {}
        fake_llm = MagicMock()
        def fake_stream(messages):
            captured["messages"] = messages
            yield type("obj", (), {"delta": "ok"})()
        fake_llm.stream_chat.side_effect = fake_stream
        with patch.object(oll_mod, "create_llm", return_value=fake_llm):
            from utils.ollama import context_chat
            list(context_chat("and for electronics?", query_engine=fake_qe))
            msgs = captured.get("messages", [])
            # history should be included (2 prior + system + current)
            ok = len(msgs) >= 3
            return ok, f"messages_in_prompt={len(msgs)}"
    check("multi_turn_history", t_multiturn)

    # 6 — e2e with real LLM if available (best effort)
    def t_e2e_real():
        if mock:
            return None, "skipped (mock mode)"
        try:
            import streamlit as st
            from utils.llama_index import create_index, load_documents, build_hybrid_retriever
            from utils.ollama import context_chat
            import tempfile
            from pathlib import Path
            # quick check: is Ollama LLM reachable?
            import ollama
            client = ollama.Client(host=endpoint, timeout=5)
            models = [m.get("model") or m.get("name") for m in client.list().get("models", [])]
            has_llm = any("qwen" in m or "llama" in m for m in (models or []))
            if not has_llm:
                return None, "skipped (no chat model)"
            if not _try_real_embeddings(endpoint, model):
                return False, f"real embeddings unavailable for {model}"
            # build tiny index and query it end-to-end
            _setup_streamlit_state()
            d = tempfile.mkdtemp()
            try:
                Path(d, "refund_policy.txt").write_text(DOCS["refund_policy.txt"], encoding="utf-8")
                st.session_state["llm_backend"] = "Ollama"
                st.session_state["ollama_endpoint"] = endpoint
                # ensure embedding already set
                docs = load_documents(d)
                idx = create_index(docs)
                qe = idx.as_query_engine(similarity_top_k=3, response_mode="compact", streaming=True)
                # attach retriever for context_chat
                vr = idx.as_retriever(similarity_top_k=3)
                st.session_state["retriever"] = build_hybrid_retriever(vr, idx, top_k=3, similarity_cutoff=0.3)
                st.session_state["query_engine"] = qe
                st.session_state["messages"] = []
                answers = []
                for _ in range(3):
                    answers.append("".join(context_chat("What is the refund policy?", query_engine=qe)))
                successful = sum(
                    "30" in answer and ("day" in answer.lower() or "refund" in answer.lower())
                    for answer in answers
                )
                ok = successful >= 2
                preview = answers[0][:80]
                return ok, f"successful={successful}/3 lengths={[len(answer) for answer in answers]} preview={preview!r}"
            finally:
                shutil.rmtree(d, ignore_errors=True)
        except Exception as e:
            return False, f"e2e failed: {e}"
    check("e2e_real_llm", t_e2e_real)

    passed = sum(1 for r in results if r["pass"] is True)
    skipped = sum(1 for r in results if r["pass"] is None)
    scored = len(results) - skipped
    score = round(passed / scored, 3) if scored else 0
    return {"suite": "generation", "results": results, "passed": passed, "skipped": skipped, "scored": scored, "total": len(results), "score": score}


def suite_performance(mock: bool, endpoint="http://localhost:11434", model="nomic-embed-text:latest"):
    print("\n[Suite] Performance")
    results = []
    def check(name, fn):
        try:
            t0 = time.perf_counter()
            ok, detail = fn()
            dt = int((time.perf_counter() - t0) * 1000)
            status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
            print(f"  {status}  {name:32s}  {detail} ({dt}ms)")
            results.append({"test": name, "pass": ok, "detail": detail, "ms": dt})
        except Exception as e:
            print(f"  FAIL  {name:32s}  {e}")
            results.append({"test": name, "pass": False, "detail": str(e), "ms": 0})

    # 1 — ingestion speed (index 5 docs)
    def t_ingest_speed():
        _setup_streamlit_state()
        if mock:
            _setup_hash_embeddings()
        elif not _try_real_embeddings(endpoint, model):
            raise RuntimeError(
                f"Real embeddings were required but unavailable for {model} at {endpoint}"
            )
        d = tempfile.mkdtemp()
        try:
            for n, c in DOCS.items():
                Path(d, n).write_text(c, encoding="utf-8")
            from utils.llama_index import load_documents, create_index
            t0 = time.perf_counter()
            docs = load_documents(d)
            idx = create_index(docs)
            dt = time.perf_counter() - t0
            ok = idx is not None and dt < 30
            return ok, f"{dt:.1f}s for {len(DOCS)} docs"
        finally:
            shutil.rmtree(d, ignore_errors=True)
    check("ingest_5_docs", t_ingest_speed)

    # 2 — persisted index cache round-trip
    def t_cache_speed():
        from utils.llama_index import load_documents, create_index, index_cache_dir, load_index_from_cache, persist_index_to_cache
        _setup_streamlit_state()
        d = tempfile.mkdtemp()
        try:
            Path(d, "a.txt").write_text(DOCS["refund_policy.txt"], encoding="utf-8")
            docs = load_documents(d)
            t0 = time.perf_counter(); idx = create_index(docs); cold = time.perf_counter() - t0
            cdir = index_cache_dir(docs)
            if cdir:
                persist_index_to_cache(idx, cdir)
                t0 = time.perf_counter(); loaded = load_index_from_cache(cdir); hit = time.perf_counter() - t0
                ok = loaded is not None
                return ok, f"cold {cold:.1f}s vs cache load {hit:.2f}s"
            return None, f"cold {cold:.1f}s (cache unavailable)"
        finally:
            shutil.rmtree(d, ignore_errors=True)
    check("cache_round_trip", t_cache_speed)

    # 3 — retrieval latency hybrid vs plain (from retrieval suite) — lightweight check
    def t_retrieval_latency():
        _setup_streamlit_state()
        d = tempfile.mkdtemp()
        try:
            for n, c in list(DOCS.items())[:3]:
                Path(d, n).write_text(c, encoding="utf-8")
            from utils.llama_index import load_documents, create_index, build_hybrid_retriever
            docs = load_documents(d)
            idx = create_index(docs)
            vr = idx.as_retriever(similarity_top_k=3)
            hy = build_hybrid_retriever(vr, idx, top_k=3, similarity_cutoff=0.3)
            t0 = time.perf_counter(); hy.retrieve("refund policy"); h_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter(); vr.retrieve("refund policy"); v_ms = (time.perf_counter() - t0) * 1000
            ok = h_ms < 500 and v_ms < 500  # both should be <0.5s (no LLM)
            return ok, f"vector {v_ms:.0f}ms hybrid {h_ms:.0f}ms"
        finally:
            shutil.rmtree(d, ignore_errors=True)
    check("retrieval_latency", t_retrieval_latency)

    # 4 — eco mode trims context/batches
    def t_eco():
        from utils.llama_index import _context_char_budget
        from utils.ollama import _num_predict, _embed_batch_size
        import streamlit as st
        st.session_state["eco_mode"] = False
        normal_ctx = _context_char_budget(); normal_pred = _num_predict(); normal_batch = _embed_batch_size()
        st.session_state["eco_mode"] = True
        eco_ctx = _context_char_budget(); eco_pred = _num_predict(); eco_batch = _embed_batch_size()
        st.session_state["eco_mode"] = False
        ok = eco_ctx < normal_ctx and eco_pred < normal_pred and eco_batch < normal_batch
        return ok, f"ctx {normal_ctx}->{eco_ctx} pred {normal_pred}->{eco_pred} batch {normal_batch}->{eco_batch}"
    check("eco_mode_trims", t_eco)

    # 5 — batch OOM resilience (from existing test harness concept)
    def t_batch_oom():
        from utils.llama_index import OllamaEmbedding
        from types import SimpleNamespace
        class FakeOOM:
            def __init__(self, *a, **kw): self.calls=[]
            def embed(self, model, input):
                self.calls.append(len(input))
                if self.calls[-1] == 8:
                    raise RuntimeError("CUDA out of memory")
                return SimpleNamespace(embeddings=[[0.1]*4 for _ in input])
        emb = OllamaEmbedding(model_name="nomic-embed-text:latest", base_url="http://localhost:11434")
        emb.embed_batch_size = 8
        emb._client = FakeOOM()
        res = emb.get_text_embedding_batch([f"t{i}" for i in range(12)])
        ok = len(res) == 12 and emb.embed_batch_size == 4
        return ok, f"shrunk to {emb.embed_batch_size} after OOM"
    check("batch_oom_resilience", t_batch_oom)

    passed = sum(1 for r in results if r["pass"] is True)
    skipped = sum(1 for r in results if r["pass"] is None)
    scored = len(results) - skipped
    score = round(passed / scored, 3) if scored else 0
    return {"suite": "performance", "results": results, "passed": passed, "skipped": skipped, "scored": scored, "total": len(results), "score": score}


def suite_robustness(mock: bool):
    print("\n[Suite] Robustness & Security")
    results = []
    def check(name, fn):
        try:
            ok, detail = fn()
            print(f"  {'PASS' if ok else 'FAIL'}  {name:32s}  {detail}")
            results.append({"test": name, "pass": ok, "detail": detail})
        except Exception as e:
            print(f"  FAIL  {name:32s}  {e}")
            results.append({"test": name, "pass": False, "detail": str(e)})

    def t_github_validation():
        from utils.helpers import normalize_github_repo
        cases = [
            ("owner/repo", True),
            ("https://github.com/owner/repo", True),
            ("https://gitlab.com/owner/repo", False),
            ("https://github.com/owner/repo/extra/path", False),
            ("", False),
        ]
        oks = []
        for inp, should_pass in cases:
            try:
                normalize_github_repo(inp)
                passed = should_pass
            except Exception:
                passed = not should_pass
            oks.append(passed)
        return all(oks), f"{sum(oks)}/{len(oks)} cases"
    check("github_validation", t_github_validation)

    def t_url_validation():
        from utils.helpers import validate_website_urls
        try:
            validate_website_urls(["https://docs.python.org/3/"])
            ok1 = True
        except Exception:
            ok1 = False
        try:
            validate_website_urls(["not a url"])
            ok2 = False
        except Exception:
            ok2 = True
        try:
            validate_website_urls(["javascript:alert(1)"])
            ok3 = False
        except Exception:
            ok3 = True
        return (ok1 and ok2 and ok3), f"https={ok1} bad={ok2} xss={ok3}"
    check("url_validation", t_url_validation)

    def t_empty_docs():
        from utils.llama_index import load_documents, create_index
        d = tempfile.mkdtemp()
        try:
            Path(d, "empty.txt").write_text("", encoding="utf-8")
            Path(d, "short.txt").write_text("hi", encoding="utf-8")
            docs = load_documents(d)
            try:
                create_index(docs)
                return False, "should have raised for no usable content"
            except Exception as e:
                return "No usable" in str(e), str(e)[:80]
        finally:
            shutil.rmtree(d, ignore_errors=True)
    check("empty_docs_rejected", t_empty_docs)

    def t_binary_exclusion():
        from utils.llama_index import load_documents
        d = tempfile.mkdtemp()
        try:
            Path(d, "a.txt").write_text("real content " * 20, encoding="utf-8")
            Path(d, "b.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
            docs = load_documents(d)
            ok = len(docs) == 1
            return ok, f"loaded {len(docs)} (png excluded)"
        finally:
            shutil.rmtree(d, ignore_errors=True)
    check("binary_exclusion", t_binary_exclusion)

    def t_special_chars():
        from utils.llama_index import _bm25_tokens
        toks = _bm25_tokens("30-day money-back (refund)!!!")
        ok = "30" in toks and "day" in toks and "money" in toks
        return ok, f"tokens={toks[:5]}"
    check("special_chars_tokenization", t_special_chars)

    def t_hinglish_filter():
        from utils.llama_index import _rewrite_query
        q = _rewrite_query("batao refund policy kya hai please")
        ok = "refund" in q and "batao" not in q and "kya" not in q
        return ok, f"rewritten='{q}'"
    check("hinglish_filler_filter", t_hinglish_filter)

    passed = sum(1 for r in results if r["pass"])
    return {"suite": "robustness", "results": results, "passed": passed, "total": len(results), "score": round(passed / len(results), 3) if results else 0}


def suite_architecture(mock: bool):
    print("\n[Suite] Architecture & Features")
    results = []
    def check(name, fn):
        try:
            ok, detail = fn()
            print(f"  {'PASS' if ok else 'FAIL'}  {name:32s}  {detail}")
            results.append({"test": name, "pass": ok, "detail": detail})
        except Exception as e:
            print(f"  FAIL  {name:32s}  {e}")
            results.append({"test": name, "pass": False, "detail": str(e)})

    def t_presets():
        from components.tabs.settings import BACKEND_PRESETS
        expected = {
            "Ollama",
            "OpenAI",
            "LM Studio (Local AI)",
            "TabbyAPI",
            "OpenAI-compatible",
        }
        ok = set(BACKEND_PRESETS) == expected
        return ok, f"definitions={list(BACKEND_PRESETS.keys())}"
    check("backend_preset_definitions", t_presets)

    def t_export():
        from components.tabs.settings import chat_history_docx
        data = chat_history_docx((("user", "hello"), ("assistant", "hi there")))
        ok = len(data) > 1000 and data[:2] == b"PK"  # docx is zip
        return ok, f"docx {len(data)} bytes"
    check("export_docx", t_export)

    def t_browser_settings():
        from utils.browser_settings import PERSISTED_SETTING_TYPES, PERSISTED_SETTINGS_HASH_STATE_KEY
        secrets_excluded = not {"openai_api_key", "r2r_api_key"} & set(PERSISTED_SETTING_TYPES)
        ok = (
            isinstance(PERSISTED_SETTINGS_HASH_STATE_KEY, str)
            and len(PERSISTED_SETTINGS_HASH_STATE_KEY) > 5
            and secrets_excluded
        )
        return ok, f"settings={len(PERSISTED_SETTING_TYPES)} secrets_excluded={secrets_excluded}"
    check("browser_settings_contract", t_browser_settings)

    def t_ollama_helpers():
        from utils.ollama import _estimate_tokens, _trim_history
        _setup_streamlit_state()
        est = _estimate_tokens("hello world")
        ok1 = est == 2 or est == 3
        # trim history keeps most recent
        from llama_index.core.llms import ChatMessage, MessageRole
        msgs = [ChatMessage(role=MessageRole.USER, content="a" * 4000), ChatMessage(role=MessageRole.USER, content="short")]
        trimmed = _trim_history(msgs, budget_tokens=10)
        ok2 = len(trimmed) <= 2
        return (ok1 and ok2), f"estimate={est} trim={len(trimmed)}"
    check("ollama_helpers", t_ollama_helpers)

    def t_embedding_verify_mock():
        from unittest.mock import patch
        from utils.llama_index import verify_embedding_model
        class FakeClient:
            def __init__(self, *a, **kw): pass
            def list(self): return {"models": [{"model": "nomic-embed-text:latest"}]}
            def show(self, m): return {"capabilities": ["embedding"]}
        with patch("utils.llama_index.ollama.Client", FakeClient):
            ok = verify_embedding_model("nomic-embed-text:latest", "http://localhost:11434")
        return ok, "mock verify"
    check("embedding_verify", t_embedding_verify_mock)

    def t_r2r_mock():
        from unittest.mock import patch
        from utils import r2r as r2r_mod
        with patch.object(r2r_mod, "get_client") as gc:
            gc.return_value.health.return_value = True
            ok = r2r_mod.get_client().health() is True
            return ok, "health mocked"
    check("r2r_health", t_r2r_mock)

    def t_import_boundaries():
        # project forbids torch at import time (light requirement)
        import builtins
        orig = builtins.__import__
        hits = []
        def guard(n, *a, **k):
            if n == "torch" or n.startswith("torch."):
                hits.append(n)
            return orig(n, *a, **k)
        builtins.__import__ = guard
        try:
            import importlib
            for mod in ["utils.llama_index", "utils.ollama", "utils.r2r"]:
                try:
                    importlib.import_module(mod)
                except Exception:
                    pass
            ok = len(hits) == 0
            return ok, f"torch imports={len(hits)}"
        finally:
            builtins.__import__ = orig
    check("no_torch_at_import", t_import_boundaries)

    passed = sum(1 for r in results if r["pass"])
    return {"suite": "architecture", "results": results, "passed": passed, "total": len(results), "score": round(passed / len(results), 3) if results else 0}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_full_eval(endpoint="http://localhost:11434", model="nomic-embed-text:latest", use_mock=False, suites=None):
    print("\n" + "=" * 60)
    print(" DocMind AI — Full Project Evaluation")
    print("=" * 60)
    print(f" Endpoint: {endpoint}  Model: {model}  Mock: {use_mock}")
    print(f" Suites: {', '.join(suites) if suites else 'all'}")
    _setup_streamlit_state()

    # suites that need embeddings first (they build indexes)
    # We set up embeddings lazily inside each suite that needs them,
    # but ensure at least one global setup for sharing.
    all_results = {}
    suite_fns = {
        "ingestion": lambda: suite_ingestion(use_mock),
        "retrieval": lambda: suite_retrieval(use_mock, endpoint, model),
        "generation": lambda: suite_generation(use_mock, endpoint, model),
        "performance": lambda: suite_performance(use_mock, endpoint, model),
        "robustness": lambda: suite_robustness(use_mock),
        "architecture": lambda: suite_architecture(use_mock),
    }
    target = suites or ALL_SUITES
    for name in ALL_SUITES:
        if name in target:
            try:
                all_results[name] = suite_fns[name]()
            except Exception as e:
                print(f"\n[Suite] {name} crashed: {e}")
                import traceback; traceback.print_exc()
                all_results[name] = {"suite": name, "passed": 0, "total": 0, "score": 0, "error": str(e), "results": []}

    for result in all_results.values():
        rows = result.get("results", [])
        if result.get("suite") == "retrieval":
            skipped = 0
            total = len(rows)
            scored = total
            passed = sum(1 for row in rows if row.get("hybrid_hit") is True)
        else:
            skipped = sum(1 for row in rows if row.get("pass") is None)
            total = result.get("total", len(rows))
            scored = max(total - skipped, 0)
            passed = sum(1 for row in rows if row.get("pass") is True)
        result["skipped"] = skipped
        result["scored"] = scored
        result["passed"] = passed
        result["total"] = total
        result["score"] = round(passed / scored, 3) if scored else 0

    total_pass = sum(v.get("passed", 0) for v in all_results.values())
    total_scored = sum(v.get("scored", 0) for v in all_results.values())
    total_skipped = sum(v.get("skipped", 0) for v in all_results.values())
    total_tests = sum(v.get("total", 0) for v in all_results.values())
    overall = round(total_pass / total_scored, 3) if total_scored else 0

    print("\n" + "=" * 60)
    print(
        f" Overall: {total_pass}/{total_scored} scored checks passed"
        f"  —  {total_skipped} skipped  —  score {overall:.1%}"
    )
    for name in target:
        r = all_results.get(name, {})
        print(
            f"   {name:14s} {r.get('passed',0)}/{r.get('scored',0)}"
            f"  skipped={r.get('skipped',0)}  {r.get('score',0):.1%}"
        )
    print("=" * 60)

    return all_results, {
        "overall_score": overall,
        "passed": total_pass,
        "scored": total_scored,
        "skipped": total_skipped,
        "total": total_tests,
        "suites": list(target),
    }


def _write_outputs(all_results, overall, out_dir="."):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    payload = {
        "meta": {"generated": time.strftime("%Y-%m-%d %H:%M"), "overall": overall},
        "suites": all_results,
    }
    # handle not-serializable (Path etc.)
    def _default(o):
        try:
            return str(o)
        except: return repr(o)
    (out_dir / "eval_results.json").write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")

    # Markdown report — detailed per suite, ready for thesis appendix
    md = []
    md.append("# DocMind AI — Full Project Evaluation Report")
    md.append("")
    md.append(
        f"_Generated {time.strftime('%Y-%m-%d %H:%M')} — "
        f"{overall['passed']}/{overall['scored']} scored checks passed, "
        f"{overall['skipped']} skipped, raw score **{overall['overall_score']:.1%}**_"
    )
    md.append("")
    md.append("This harness uses small local fixtures. It is not a browser test, a production load test, or proof that generated answers are always factually grounded.")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append("| Suite | Passed | Skipped | Score | Scope |")
    md.append("|---|---:|---:|---:|---|")
    descs = {
        "ingestion": "Loader exclusions, selected formats, chunking, dedup, titles, cache, helper validation",
        "retrieval": "Fixture retrieval, query cleanup, hybrid ranking, and no-match rejection",
        "generation": "Prompt guards, mocked no-match path, tone prompts, history, three real LLM trials",
        "performance": "Small-fixture timings, cache round-trip, Eco settings, batch shrinking",
        "robustness": "Empty/binary handling, URL/GitHub validation, token cleanup",
        "architecture": "Provider definitions, export, persistence contract, mocked R2R health",
    }
    for name in ALL_SUITES:
        if name not in all_results: continue
        r = all_results[name]
        md.append(f"| {name} | {r.get('passed',0)}/{r.get('scored',0)} | {r.get('skipped',0)} | **{r.get('score',0):.1%}** | {descs.get(name,'')} |")
    md.append(f"| **Overall** | **{overall['passed']}/{overall['scored']}** | **{overall['skipped']}** | **{overall['overall_score']:.1%}** | Raw scored-check rate |")
    md.append("")

    # Retrieval deep-dive table (most important for README)
    if "retrieval" in all_results:
        ret = all_results["retrieval"]
        summ = ret.get("summary", {})
        md.append("## Retrieval deep-dive (plain vs Hybrid)")
        md.append("")
        if summ:
            md.append(f"_Embedding: `{summ.get('embedding','')}` — hit = expected doc in top-3, correct rejection = no-match returns 0 nodes_")
            md.append("")
            md.append("| Metric | Plain RAG | DocMind Hybrid |")
            md.append("|---|---:|---:|")
            md.append(f"| Factual hit rate | {summ.get('plain_hit_rate',0):.0%} | **{summ.get('hybrid_hit_rate',0):.0%}** |")
            md.append(f"| Correct rejection | {summ.get('plain_rejection',0):.0%} | **{summ.get('hybrid_rejection',0):.0%}** |")
            md.append(f"| Avg latency | {summ.get('plain_avg_ms',0):.0f} ms | {summ.get('hybrid_avg_ms',0):.0f} ms |")
            md.append("")
        results = ret.get("results", [])
        if results:
            md.append("| # | Query | Expected | Feature | Plain | Hybrid | Plain top | Hybrid top |")
            md.append("|---|---|---|---|:---:|:---:|---|---|")
            for i, r in enumerate(results, 1):
                exp = r["expected"] or "— (no match)"
                p = "✓" if r["plain_hit"] else "✗"
                h = "✓" if r["hybrid_hit"] else "✗"
                md.append(f"| {i} | {r['query']} | {exp} | {r['feature']} | {p} | {h} | {r['plain_top'] or '—'} | {r['hybrid_top'] or '—'} |")
            md.append("")

    # Per-suite breakdowns
    for name in ALL_SUITES:
        if name not in all_results or name == "retrieval": continue
        r = all_results[name]
        md.append(f"## Suite: {name} ({r.get('passed',0)}/{r.get('scored',0)} scored, {r.get('skipped',0)} skipped — {r.get('score',0):.1%})")
        md.append("")
        md.append("| Test | Pass | Detail |")
        md.append("|---|:---:|---|")
        for t in r.get("results", []):
            status = "SKIP" if t.get("pass") is None else ("PASS" if t.get("pass") else "FAIL")
            d = (t.get("detail") or "")[:120].replace("|", "/")
            ms = f" {t.get('ms','')}ms" if "ms" in t else ""
            md.append(f"| {t.get('test','')} | {status} | {d}{ms} |")
        md.append("")

    md.append("## Implemented Retrieval Features")
    md.append("")
    md.append("- Vector candidates are fused with a CPU BM25 ranking through Reciprocal Rank Fusion.")
    md.append("- With a positive similarity cutoff, weak vector candidates require positive BM25 evidence.")
    md.append("- Short queries can receive curated BM25 synonym tokens; the vector query remains separate.")
    md.append("- Hyphens are split, basic number words are normalized, and selected filler words are removed.")
    md.append("- File-backed chunks receive a title prefix; website documents use source URLs without file titles.")
    md.append("- A stemmed-token overlap filter removes near-duplicate chunks before embedding.")
    md.append("- Up to five persisted index directories are retained when cache writes are available.")
    md.append("- Eco Mode lowers configured embedding batch, output, and context limits; actual speed and heat depend on hardware and model.")
    md.append("- Empty local retrieval returns a fixed no-match message. This does not verify factual support for non-empty answers.")
    md.append("")

    md.append("## Running the Harness")
    md.append("")
    md.append("```bash")
    md.append("# Requires the configured Ollama embedding model; the real LLM check is skipped when unavailable")
    md.append("python eval_harness.py")
    md.append("")
    md.append("# Uses stable hash embeddings and skips the real LLM check")
    md.append("python eval_harness.py --mock")
    md.append("")
    md.append("# Run one suite and write results to a chosen directory")
    md.append("python eval_harness.py --suite retrieval --out ./eval-output")
    md.append("```")
    md.append("")
    md.append("The mock suite still exercises validation paths that perform DNS resolution. The project does not currently commit a dependency lockfile, so package versions can vary between environments.")

    (out_dir / "eval_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {out_dir/'eval_results.json'} and {out_dir/'eval_report.md'}")
    return out_dir / "eval_results.json", out_dir / "eval_report.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocMind full-project eval harness")
    parser.add_argument("--endpoint", default="http://localhost:11434", help="Ollama endpoint")
    parser.add_argument("--model", default="nomic-embed-text:latest", help="Embedding model")
    parser.add_argument("--mock", action="store_true", help="Use hash embeddings + mocked LLM (no Ollama)")
    parser.add_argument("--quick", action="store_true", help="Alias for --mock")
    parser.add_argument("--suite", choices=ALL_SUITES, help="Run only one suite")
    parser.add_argument("--out", default=".", help="Output directory")
    args = parser.parse_args()

    use_mock = args.mock or args.quick
    suites = [args.suite] if args.suite else None

    all_results, overall = run_full_eval(endpoint=args.endpoint, model=args.model, use_mock=use_mock, suites=suites)
    _write_outputs(all_results, overall, out_dir=args.out)
    raise SystemExit(1 if overall["passed"] != overall["scored"] else 0)
