# Literature Review Notes

Date: 2026-05-28

Purpose: collect the outside research and official library/API references used to design the BetterBasket product-matching system.

## Local Reference Files

Downloaded into this directory:

- `entity_resolution_blocking_survey_1905.06167.pdf`
  - Source: https://arxiv.org/abs/1905.06167
  - Use: entity-resolution blocking/filtering patterns, especially schema-agnostic blocking when source schemas differ.
- `deepmatcher_sigmod18.pdf`
  - Source: https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf
  - Use: neural entity matching framing and limits.
- `ditto_vldb2020.pdf`
  - Source: https://arxiv.org/abs/2004.00584 and https://www.vldb.org/pvldb/vol14/p50-li.pdf
  - Use: entity matching with pretrained language models, domain knowledge injection, summarizing long strings, and difficult-example augmentation.
- `sentence_bert_1908.10084.pdf`
  - Source: https://arxiv.org/abs/1908.10084
  - Use: dense sentence embeddings for scalable semantic similarity.
- `faiss_billion_scale_similarity_1702.08734.pdf`
  - Source: https://arxiv.org/abs/1702.08734
  - Use: approximate/exact nearest-neighbor scaling if future datasets outgrow sparse TF-IDF search.

## Official Library and API References

- scikit-learn `TfidfVectorizer`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- RapidFuzz process API: https://rapidfuzz.github.io/RapidFuzz/Usage/process.html
- Sentence Transformers semantic textual similarity docs: https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
- FAISS repository/docs: https://github.com/facebookresearch/faiss
- OpenAI GPT-5.4 nano model page: https://developers.openai.com/api/docs/models/gpt-5.4-nano
- OpenAI Responses API reference: https://developers.openai.com/api/reference/resources/responses/methods/create
- OpenAI Structured Outputs guide: https://developers.openai.com/api/docs/guides/structured-outputs
- OpenAI Batch API guide: https://developers.openai.com/api/docs/guides/batch

## Synthesis For BetterBasket

Entity-resolution systems should avoid all-pairs comparison by generating candidate sets, then scoring candidates with richer features. The survey literature supports blocking/filtering, but our local EDA shows exact schema-aligned blocking is too narrow because A and B category taxonomies differ and A has sparse brand/size coverage.

Deep entity-matching work supports pairwise reranking with learned or language-model features, but it also implies we need careful candidate retrieval first. A model cannot be asked to compare all A x B products. The local catalog has about 12.9 billion possible pairs, so retrieval and blocking are non-negotiable.

SBERT-style embeddings are promising for semantic fallback because they produce embeddings once and compare them with cosine similarity. The venv already has `sentence_transformers`, but no ANN library such as FAISS or HNSW is installed. Therefore dense retrieval should be an optional second-pass experiment for this assessment, not the first dependency-heavy path.

The official scikit-learn and local EDA both support TF-IDF as the first universal retrieval strategy. The second-pass probe measured about 14 minutes for full top-10 retrieval over A against B, which is acceptable for this assessment and scalable enough for several competitor CSVs if matrices are cached per store.

RapidFuzz remains the right candidate-level string matcher. Its process API supports efficient best-match extraction and score cutoffs; we should use it after candidate generation rather than as a global search over all products.

OpenAI GPT-5.4 nano is explicitly positioned for classification, extraction, ranking, and sub-agent tasks, with structured outputs, function calling, Responses API support, and Batch API support. That makes it a good constrained judge for borderline product pairs or small candidate sets, not a replacement for deterministic retrieval and scoring.

The Batch API is relevant because final ambiguity review can be asynchronous. It offers lower cost, higher rate-limit headroom, and a 24-hour turnaround model, which fits offline product matching.

