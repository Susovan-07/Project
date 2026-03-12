# Post Snippet Similarity Matcher
## AI/ML Hiring Assignment — Full Production Pipeline

## How to Run

### Minimal (no installs needed — runs immediately)
    python3 post_similarity_matcher.py
    python3 post_similarity_matcher.py path/to/your_file.xlsx

### Full pipeline (install extras for best results)
    pip install sentence-transformers faiss-cpu anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 post_similarity_matcher.py

---

## Pipeline — What Each Package Unlocks

Without anything installed (works now):
  Stage 1  : BM25 from scratch (k1=1.5, b=0.75)
  Stage 2  : LSA semantic embeddings (char ngrams + TruncatedSVD 200-dim)
  ANN      : sklearn NearestNeighbors (exact cosine, all CPU cores)
  Titles   : Keyword extraction (top-3 terms per cluster)

With sentence-transformers installed:
  Stage 2  : paraphrase-multilingual-MiniLM-L12-v2 neural embeddings
             (true semantic understanding, handles Hindi+English+Hinglish natively)

With faiss-cpu installed:
  ANN      : FAISS IndexFlatIP (SIMD-optimized, scales to 100M vectors)

With ANTHROPIC_API_KEY set:
  Titles   : Claude API generates human-readable English topic titles per cluster

---

## Architecture

    Excel File
        |
    [STEP 1] Load all sheets (openpyxl read_only — memory efficient)
        |
    [STEP 2] Multilingual Preprocessing
             Unicode NFC → remove URLs/emojis/mentions → hashtag→word
             → selective lowercase → Hindi+Hinglish+English stopwords
        |
    [STAGE 1] BM25 Candidate Filtering (from scratch)
              Scores all 1,774 posts against each other
              Returns top-50 candidates per post
              Reduces 3.1M pairs → 88K candidates (2.8%)
        |
    [STAGE 2] Semantic Embedding + ANN Search
              LSA: char ngrams → TF-IDF → TruncatedSVD → 200-dim vectors
              OR: SentenceTransformer → 384-dim neural vectors
              ANN: sklearn NearestNeighbors OR FAISS IndexFlatIP
              Reranks candidates by true semantic similarity
        |
        +---> [COMPARE] TF-IDF + cosine similarity
        +---> [BASELINE] Fuzzy matching on 150-record sample
        |
    [CLUSTER] Agglomerative clustering on semantic embeddings
              Cosine distance, average linkage
        |
    [TITLES] Claude API OR keyword extraction
        |
    Excel Output (6 sheets)

---

## Output Sheets

1. Similarity_Matches     — Top-5 matches per post with score and method
2. Cluster_Assignments    — Every post with cluster_id, cluster_title, cluster_usernames
3. Cluster_Summary        — One row per cluster: size, usernames, platforms, sample post
4. Method_Comparison      — Semantic vs TF-IDF vs Fuzzy side-by-side with analysis
5. Fuzzy_Baseline         — Raw fuzzy results on 150-record sample
6. Pipeline_Info          — Config used, capabilities active, stats

---

## Configuration (edit CFG dict at top of script)

    bm25_k1            = 1.5    TF saturation
    bm25_b             = 0.75   Length normalization
    bm25_candidates    = 50     BM25 candidates passed to Stage 2
    semantic_threshold = 0.55   Min score for final output
    top_n              = 5      Matches per post
    cluster_size       = 10     Target posts per cluster
    llm_max_clusters   = 40     Max clusters sent to LLM

---

## Files

    post_similarity_matcher.py   — Main script
    APPROACH_NOTE.md             — Full methodology explanation
    README.md                    — This file
    post_similarity_results.xlsx — Output
