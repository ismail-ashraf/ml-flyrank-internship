# RAG System Optimization: Chunking Strategies & Vector Retrieval Reranking
### Structured Analysis Report — ML Engineering Pipeline

---

## Overview

### Problem Statement

Retrieval-Augmented Generation (RAG) systems suffer from a fundamental tension: LLMs have bounded context windows, but knowledge corpora are unbounded. Naïve approaches (fixed-size chunking + cosine-similarity retrieval) fail at production scale due to three compounding failure modes:

1. **Semantic fragmentation** — splitting mid-concept destroys embedding coherence.
2. **Recall-precision imbalance** — dense retrieval maximizes recall but surfaces noisy, low-precision chunks.
3. **Context dilution** — stuffing the LLM prompt with weakly-relevant chunks degrades generation quality.

### Novel Contributions Covered

| Area | Technique | Key Gain |
|---|---|---|
| Chunking | Semantic + Hierarchical splitting | +18–34% retrieval relevance vs. fixed-size |
| Retrieval | Hybrid dense-sparse (BM25 + bi-encoder) | +12–20% recall@10 |
| Reranking | Cross-encoder reranking (ColBERT / MonoT5) | +15–28% MRR@10 |
| Fusion | Reciprocal Rank Fusion (RRF) | Robust multi-signal merging |

### Datasets & Metrics

- **Benchmarks**: BEIR, MS-MARCO, NaturalQuestions, TriviaQA
- **Metrics**: Recall@K, MRR@10, NDCG@10, Answer Faithfulness (RAGAs), Context Relevance Score
- **Baselines**: Fixed-size chunking + FAISS flat-L2, BM25 alone, bi-encoder alone

---

## Technical Core

### STEP 1 — Chunking Strategies

#### A. Fixed-Size Chunking (Baseline)
Split document every `N` tokens with stride `S`. Simple but semantically destructive.

```
chunk_i = tokens[i*S : i*S + N]
```

**Failure**: A sentence like *"mitochondria produce ATP via oxidative phosphorylation — a process requiring..."* split at token 512 loses the predicate.

#### B. Semantic Chunking (Embedding-Based Boundary Detection)
Compute sentence embeddings, detect breakpoints where cosine similarity drops below threshold `θ`.

```
sim(s_i, s_{i+1}) = cos(E(s_i), E(s_{i+1}))
breakpoint if sim < θ  (typically 0.75–0.85)
```

**Advantage**: Preserves topical coherence. 20–30% larger chunks on average but semantically complete.

#### C. Hierarchical / Parent-Child Chunking
Index **small child chunks** for retrieval precision; return **large parent chunk** to the LLM for context richness.

```
parent_chunk  = 1024 tokens  (fed to LLM)
child_chunk   = 128  tokens  (indexed in vector DB)
child.parent_id → parent_chunk retrieval
```

This is the **SmallToLarge** pattern. RAPTOR extends this with recursive summarization trees.

#### D. Late Chunking (ColBERT-style)
Encode the full document first, then pool token embeddings per chunk — preserving cross-chunk attention context.

```
token_embeddings = Encoder(full_doc)   # [L, d]
chunk_emb_i      = mean(token_embeddings[start_i:end_i])
```

---

### STEP 2 — Hybrid Retrieval

#### Dense Retrieval (Bi-Encoder)
```
score_dense(q, d) = cos(E_q(q), E_d(d))
```
- Fast: pre-computed doc embeddings, FAISS ANN search
- Weakness: lexical mismatch (e.g., "myocardial infarction" vs. "heart attack")

#### Sparse Retrieval (BM25)
```
BM25(q, d) = Σ_{t∈q} IDF(t) · (TF(t,d) · (k₁+1)) / (TF(t,d) + k₁·(1 - b + b·|d|/avgdl))
```
- Strong on exact-match / rare terms
- Weakness: semantic gap

#### Reciprocal Rank Fusion (RRF)
Merge dense and sparse ranked lists without score normalization:

```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))
```
- `k = 60` (standard constant to dampen high-rank outlier inflation)
- Works across incommensurable score scales — no normalization needed

---

### STEP 3 — Reranking

#### Cross-Encoder Reranking
A cross-encoder jointly encodes `(query, document)` pairs — capturing full cross-attention:

```
score = CrossEncoder([CLS] query [SEP] chunk [SEP])
```

- **MonoT5**: T5 fine-tuned on MS-MARCO to output "true"/"false" token logits
- **ColBERT**: Late-interaction model — token-level MaxSim scoring

```
# ColBERT MaxSim
score(q, d) = Σ_{i∈q} max_{j∈d} cos(q_i, d_j)
```

**Latency trade-off**: Cross-encoders are O(N·L²) vs. bi-encoder O(1) at query time. Only feasible on a small rerank candidate set (top-50 to top-200).

#### Loss Functions for Fine-Tuning Rerankers

**Pairwise ranking loss (RankNet)**:
```
L = -log σ(s_+ − s_−)
```

**ListMLE (listwise)**:
```
L = -log P(π* | s) = -Σ_i log [exp(s_{π*(i)}) / Σ_{j≥i} exp(s_{π*(j)})]
```

---

## Mathematical Summary — Full Pipeline

```
1. q  →  [BM25 top-K]  +  [ANN Dense top-K]          # Hybrid Retrieval
2. candidates  →  RRF fusion                            # Score Fusion
3. top-M candidates  →  CrossEncoder(q, c_i)           # Reranking
4. top-N reranked chunks  →  LLM(q, context)           # Generation
```

**Complexity**:
- BM25: O(|vocab| + |docs|) index, O(|q|·|d|) query
- ANN (HNSW): O(log N) query, O(N·d) index
- Reranker: O(M · L²) — M=candidates, L=sequence length (bottleneck)

---

## Pseudocode

```python
# ============================================================
# RAG Pipeline: Chunking + Hybrid Retrieval + Reranking
# PyTorch / HuggingFace / FAISS Implementation Blueprint
# ============================================================

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

# ── CONFIG ──────────────────────────────────────────────────
BI_ENCODER_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE        = 256        # tokens, child chunk
PARENT_SIZE       = 1024       # tokens, parent chunk
SEMANTIC_THRESH   = 0.78       # cosine similarity breakpoint
DENSE_TOP_K       = 100        # initial dense retrieval
BM25_TOP_K        = 100        # initial sparse retrieval
RRF_K             = 60         # RRF constant
RERANK_TOP_M      = 50         # candidates sent to reranker
FINAL_TOP_N       = 5          # chunks injected into LLM prompt


# ── STEP 1: HIERARCHICAL SEMANTIC CHUNKING ──────────────────

def semantic_chunk(text: str, tokenizer, threshold=SEMANTIC_THRESH):
    """Split text at semantic breakpoints using embedding similarity."""
    sentences = text.split(". ")
    bi_enc = SentenceTransformer(BI_ENCODER_MODEL)
    embeddings = bi_enc.encode(sentences, convert_to_tensor=True)

    breakpoints = [0]
    for i in range(len(sentences) - 1):
        sim = torch.nn.functional.cosine_similarity(
            embeddings[i].unsqueeze(0), embeddings[i+1].unsqueeze(0)
        ).item()
        if sim < threshold:
            breakpoints.append(i + 1)
    breakpoints.append(len(sentences))

    chunks = []
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        chunks.append(" ".join(sentences[start:end]))
    return chunks


def build_parent_child_index(documents: list[str], tokenizer):
    """
    Build hierarchical index:
    - child_chunks: small (indexed for retrieval)
    - parent_map:   child_id -> parent_chunk (returned to LLM)
    """
    child_chunks, parent_map = [], {}
    for doc_id, doc in enumerate(documents):
        # Build parent chunks (large)
        parent_tokens = tokenizer.encode(doc)
        parents = [
            tokenizer.decode(parent_tokens[i:i+PARENT_SIZE])
            for i in range(0, len(parent_tokens), PARENT_SIZE)
        ]
        # Build child chunks (small, semantically split)
        for p_idx, parent in enumerate(parents):
            children = semantic_chunk(parent, tokenizer)
            for child in children:
                cid = len(child_chunks)
                child_chunks.append(child)
                parent_map[cid] = parent     # child → parent lookup
    return child_chunks, parent_map


# ── STEP 2: HYBRID INDEX CONSTRUCTION ──────────────────────

class HybridRetriever:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.bi_enc = SentenceTransformer(BI_ENCODER_MODEL)

        # Dense index (FAISS)
        embeddings = self.bi_enc.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        # Sparse index (BM25)
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = DENSE_TOP_K):
        # Dense retrieval
        q_emb = self.bi_enc.encode([query], convert_to_tensor=False)
        q_emb = np.array(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)
        _, dense_ids = self.index.search(q_emb, k)
        dense_ranks = {doc_id: rank for rank, doc_id in enumerate(dense_ids[0])}

        # Sparse retrieval
        scores = self.bm25.get_scores(query.lower().split())
        sparse_ids = np.argsort(scores)[::-1][:BM25_TOP_K]
        sparse_ranks = {doc_id: rank for rank, doc_id in enumerate(sparse_ids)}

        # RRF Fusion
        all_ids = set(dense_ranks) | set(sparse_ranks)
        rrf_scores = {}
        for doc_id in all_ids:
            dr = dense_ranks.get(doc_id, k + 1)
            sr = sparse_ranks.get(doc_id, BM25_TOP_K + 1)
            rrf_scores[doc_id] = 1/(RRF_K + dr) + 1/(RRF_K + sr)

        fused = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return fused[:RERANK_TOP_M]


# ── STEP 3: CROSS-ENCODER RERANKING ────────────────────────

class Reranker:
    def __init__(self):
        self.cross_enc = CrossEncoder(CROSS_ENCODER_MODEL)

    def rerank(self, query: str, candidate_ids: list[int],
               chunks: list[str], parent_map: dict, top_n=FINAL_TOP_N):
        pairs = [(query, chunks[cid]) for cid in candidate_ids]
        scores = self.cross_enc.predict(pairs)
        ranked = sorted(zip(candidate_ids, scores),
                        key=lambda x: x[1], reverse=True)
        top_ids = [cid for cid, _ in ranked[:top_n]]
        # Return parent chunks for LLM context richness
        return [parent_map.get(cid, chunks[cid]) for cid in top_ids]


# ── STEP 4: FULL PIPELINE QUERY ────────────────────────────

def rag_query(query: str, retriever: HybridRetriever,
              reranker: Reranker, parent_map: dict, llm_fn):
    # 1. Hybrid retrieval → fused candidate list
    candidate_ids = retriever.retrieve(query)

    # 2. Cross-encoder reranking → top-N parent chunks
    context_chunks = reranker.rerank(
        query, candidate_ids, retriever.chunks, parent_map
    )

    # 3. Construct LLM prompt
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""Answer the question using only the provided context.

Context:
{context}

Question: {query}
Answer:"""

    return llm_fn(prompt)
```

---

## Production Considerations

### Failure Points & Mitigations

| Risk | Cause | Mitigation |
|---|---|---|
| **Semantic boundary misdetection** | threshold `θ` too high/low | Calibrate per-domain; use sliding window fallback |
| **Embedding drift** | Bi-encoder trained on different distribution than your corpus | Fine-tune on domain-specific query–passage pairs (hard-negative mining) |
| **Reranker latency spike** | Cross-encoder O(M·L²) at query time | Async reranking; cap M at 50; use distilled cross-encoder |
| **FAISS index memory** | 768-dim × 10M docs ≈ 28 GB | Use IVF-PQ quantization; PQ reduces to 2–4 GB at <2% recall loss |
| **BM25 vocabulary explosion** | Medical / code corpora with rare tokens | Stemming + subword BM25 variant |
| **Parent chunk over-retrieval** | Multiple children map to same parent → duplicate context | Dedup parent chunks before LLM injection |
| **RRF parameter sensitivity** | k=60 default not always optimal | Tune k on validation set; typical range: 30–100 |
| **Stale index** | Document corpus updates not reflected | Incremental FAISS + BM25 index update pipeline with versioning |

### Scaling Architecture

```
Query
  │
  ├─► BM25 Service (Elasticsearch / OpenSearch)
  │                    ↓
  └─► ANN Service  (FAISS / Weaviate / Qdrant / Pinecone)
                       ↓
              RRF Fusion Layer (in-process)
                       ↓
              Reranker Service (GPU microservice)
                       ↓
              Context Assembly → LLM API
```

**Recommended stack**: LangChain / LlamaIndex for orchestration, Qdrant for hybrid (dense + sparse native), vLLM for LLM serving, FastAPI for reranker microservice.

### Monitoring Signals

- `retrieval_recall@5` — compare retrieved chunks against labeled ground truth
- `context_relevance_score` — RAGAs metric: LLM-judged relevance of injected context
- `answer_faithfulness` — hallucination detection
- `reranker_latency_p99` — alert if >200ms; scale GPU replicas
- `chunk_length_distribution` — detect chunker regression after corpus updates

---

*Report generated via 3-step ML Engineering Analysis Pipeline.*
