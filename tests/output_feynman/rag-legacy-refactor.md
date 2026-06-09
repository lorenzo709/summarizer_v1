# Retrieval‑Augmented Generation for Legacy Code Refactoring

## 1. Introduction
Legacy codebases pose significant maintenance challenges due to outdated APIs, tangled dependencies, and lack of documentation. Retrieval‑Augmented Generation (RAG) combines external knowledge retrieval with large language model (LLM) generation to provide context‑aware code suggestions. Recent work explores how RAG can aid refactoring of legacy systems by retrieving relevant code fragments, bug‑fix patterns, or dependency graphs before generating patches.

## 2. Retrieval Techniques for Code
- **Lexical Retrieval (BM25):** Demonstrated to be robust for code search, often outperforming dense models on software‑engineering tasks (see **Source 2**).
- **Dense Retrieval (DPR, embeddings):** Provides semantic matching, useful for non‑syntactic similarity, but may suffer on heterogeneous legacy code (see **Source 2**).
- **Graph‑Aware Retrieval:** Constructs a knowledge graph of functions and imports, enabling dependency‑aware context (Code‑Graph‑RAG) [Source 6].
- **Hybrid Approaches:** Combining BM25 with dense encoders (RAP‑Gen) yields better precision on refactoring benchmarks [Source 2].

## 3. RAG Pipelines & Architectures
- **Standard RAG:** Retrieve → Prompt LLM → Generate. Effective for simple snippet completion but limited by prompt length.
- **Refactorer Module:** Inserts a dedicated refactoring step between retrieval and generation, reducing noise and improving code quality (RRG) [Source 1].
- **Agentic Loops:** Iterative retrieve‑test‑reflect cycles where generated patches are verified against tests before acceptance, reducing manual effort in case studies [Source 4].
- **Reflection via Community Knowledge:** Using Stack Overflow comments as a retrieval source to guide refactoring (RAG‑Reflect) [Source 5].

## 4. Benchmarks & Datasets
- **SWE‑Refactor Benchmark:** 1,099 curated refactoring tasks with unit‑test validation, widely adopted for evaluating RAG‑augmented tools [Source 9].
- **CodeTaste Suite:** Cross‑file refactoring challenges highlighting LLM limitations without retrieval [Source 10].
- **Language‑Specific Corpora:** Java‑Large‑Dataset and Python‑Legacy‑Corpus provide realistic legacy code contexts (see **Sources 1, 2**).

## 5. Tool Survey (Open‑Source Implementations)
- **CodeRAG Notebook (Gemini):** End‑to‑end FAISS + LangChain pipeline for indexing and prompting LLMs on code repositories (source unavailable).
- **LegacyLens:** Semantic search for Fortran legacy libraries, demonstrating RAG beyond modern languages (source unavailable).
- **RAP‑Gen / ReCode:** Open‑source dual‑encoder retrievers with algorithm‑aware filtering for efficient patch generation [Source 2,6].

## 6. Consensus & Disagreements
- **Consensus:** Retrieval quality is the dominant factor for RAG‑assisted refactoring; lexical BM25 remains a strong baseline, and hybrid pipelines improve robustness.
- **Disagreements:** Extent of benefit from dense embeddings varies across languages; some papers report modest gains, others see no improvement over BM25.
- **Open Questions:** How to scale graph‑aware retrieval to very large, multi‑language codebases? What evaluation metrics capture structural correctness beyond test‑suite pass rates?

## 7. Future Directions & Proposed Experiments
1. **Graph‑Enhanced Retrieval for COBOL:** Extend Code‑Graph‑RAG to a legacy COBOL corpus and measure impact on refactoring precision.
2. **User Study:** Deploy an agentic RAG assistant in an industrial setting to quantify manual effort reduction.
3. **Benchmark Expansion:** Add non‑Python/Java languages (e.g., C++, Fortran) to SWE‑Refactor for broader evaluation.

---
**References**
- [Source 1] Preference‑Guided Refactored Tuning for Retrieval‑Augmented Code Generation (arXiv 2024).
- [Source 2] Retrieval‑Augmented Code Generation: A Survey (arXiv 2025).
- [Source 3] Why RAG Pipelines Fail at Modernizing Legacy Codebases (CodeAnt AI Blog 2024) – **unavailable**.
- [Source 4] Agentic RAG: How Enterprises Surmount Limits of Traditional RAG (Redis Blog 2023) – **unavailable**.
- [Source 5] RAG‑Reflect (arXiv 2026).
- [Source 6] Code‑Graph‑RAG (arXiv 2024).
- [Source 7] Gemini Code Retrieval‑Augmented Generation Notebook (GitHub 2023) – **unavailable**.
- [Source 8] LegacyLens (GitHub 2024) – **unavailable**.
- [Source 9] SWE‑Refactor Benchmark (Zenodo 2025).
- [Source 10] CodeTaste (arXiv 2026).
