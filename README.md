# RAG Assignment: Baseline vs. Re-Ranking

Two retrieval pipelines over the same chunked document, compared on hard queries:

| | Pipeline 1 – Baseline | Pipeline 2 – Re-Ranking |
|---|---|---|
| Retrieval | bi-encoder (`all-MiniLM-L6-v2`) + FAISS, Top 10 by cosine similarity | same Top 10 |
| Re-ranking | – | cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores all 10 (query, chunk) pairs |
| LLM context | first 3 by cosine | first 3 by cross-encoder score |
| Generation | Gemini `gemini-2.5-flash`, shared prompt → Answer 1 | same prompt → Answer 2 |

**Document:** Regulation (EU) 2016/679 – the General Data Protection Regulation (GDPR), official EUR-Lex PDF
(88 pages, 173 recitals + 99 articles → 565 chunks). EU legislation published on EUR-Lex may be reused
([Commission Decision 2011/833/EU](https://eur-lex.europa.eu/eli/dec/2011/833/oj)).

All code is in **[`RAG_Reranking.ipynb`](RAG_Reranking.ipynb)**; it is committed with its outputs.

## Repository layout

```
RAG_Reranking.ipynb          main notebook (all code, commented, with outputs)
data/GDPR_EU_2016_679.pdf    source document
results/                     CSV tables produced by the notebook + LLM answer cache
  failure_summary.csv          rank shifts and answer verdicts for the failure queries
  candidate_pool_ranks.csv     gold-chunk ranks for all 109 candidate queries
  candidate_pool_summary.csv   Recall@10 / Hit@3 / MRR@10 per query type
  latency_stages.csv           per-stage latency (mean / median / p95)
  latency_pipelines.csv        end-to-end latency of both pipelines
  scaling_extrapolation.csv    cross-encoder-on-everything vs. retrieve-then-rerank
  llm_cache.json               Gemini answers + measured latency, keyed by prompt hash
```

The written report is in [`report/RAG_Assignment_Report.pdf`](report/RAG_Assignment_Report.pdf).

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # put your GEMINI_API_KEY in .env
jupyter notebook RAG_Reranking.ipynb
```

- Runs on CPU; the two MiniLM models are downloaded from Hugging Face on first use.
- Gemini answers are cached in `results/llm_cache.json`, so the notebook re-runs with the same answers and
  **without an API key**. Set `USE_LLM_CACHE = False` in section 6 to call the API again.

## Notebook contents

1. Setup – configuration, hardware info
2. Loading the PDF – font-aware cleaning (drops footers / footnotes, rejoins words split by soft hyphens), split into recitals and articles
3. Chunking – ≤150 words, 30-word overlap, never crossing an article; checked against the 256-token encoder limit
4. Pipeline 1 – bi-encoder embeddings, FAISS `IndexFlatIP`, Top 10 → Top 3
5. Pipeline 2 – cross-encoder re-ranking of the same Top 10 → Top 3
6. Answer generation – one prompt template shared by both pipelines
7. Failure mode analysis – keyword trap, exception, negation and a document-specific case, plus limitation cases
8. Summary – rank-shift table, answer verdicts, Hit@3 / MRR@10 over the whole candidate pool
9. Latency profiling – per-stage and end-to-end timings
10. Scaling experiment – cross-encoder over the full corpus vs. retrieve-then-rerank
