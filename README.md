# Lightweight RAG System for Academic Papers


> **A retrieval-augmented generation (RAG) system that answers questions about research papers using sub-2GB models on CPU.**

Combines adaptive section-aware chunking, hybrid dense-sparse retrieval (BM25 + semantic embeddings), and systematic evaluation across 5 LLMs. Designed for resource-constrained environments.

[**Demo**](#demo) | [**Quick Start**](#quick-start) | [**Architecture**](#architecture) 

---

## Why This Project?

Systematic literature reviews and meta-analyses require extracting specific information from several research papers—treatment effectiveness, disease prevalence, sample characteristics, etc. Global disease burden estimation studies rely heavily on manually extracting data from published literature.

This system was initially designed to streamline that data extraction workflow. With advances in large language models, it has evolved beyond simple extraction to provide comprehensive question answering: summarizing findings, synthesizing information across sections, and generating concise responses grounded in retrieved evidence.

The transparency of showing retrieved chunks alongside answers serves two purposes:
1. **Verification**: Users can validate that relevant context was found
2. **Query refinement**: Inspecting retrieved chunks helps users rephrase questions for better retrieval

### Current Limitations

- **Chunking**: Fixed-size boundaries may split tables, equations, or complex figures across chunks, losing context
- **LLM constraints**: Models <2GB occasionally struggle with complex multi-hop reasoning across chunks
- **Single-document focus**: Currently optimized for querying one paper at a time; multi-document comparison not yet supported


**Addresses:**
- Runs on CPU with models <2GB (LongT5, Flan-T5)
- Section-aware adaptive chunking respects document structure
- Hybrid retrieval combines semantic search + keyword matching
- Return answer plus citations (chunk IDs / sections / positions)
- Empirical comparison of 5 LLMs, 3 retrieval methods, 2 chunking strategies

---

## Key Features

### Intelligent Document Processing
- **PDF artifact removal**: Fixes hyphenation (`method-\nology` → `methodology`), line breaks, boilerplate
- **Adaptive chunking**: Respects paragraph/sentence boundaries (200-1000 chars, target: 600)
- **Section-aware parsing**: Automatically detects Methods, Results, Discussion sections in academic papers

### Hybrid Retrieval System
- **Dense retrieval**: Multi-QA MPNet embeddings (768-dim) with cosine similarity
- **Sparse retrieval**: BM25 for exact keyword matching
- **Fusion strategies**: Reciprocal Rank Fusion (RRF) or weighted score combination
- **Smart truncation**: Relevance-based context selection when token limits exceeded


---

## Demo

### Web Interface (Gradio)
```bash
python apps/gradio_app.py
```

![Demo Screenshot](assets/demo_screenshot.png)

### Example Interaction

**Question:** *"What methods were used to collect data in this study?"*

**Retrieved Chunks:**
1. Chunk 23 (Methods section, score: 0.94)
2. Chunk 21 (Sample size details, score: 0.87)
3. Chunk 19 (Data collection timeline, score: 0.81)

**Generated Answer:**
> "The study employed cross-sectional household surveys conducted in southern Syria between June 2016 and February 2017. The sample size was calculated to be 87-96 households per sub-district, using structured interviews and water quality testing for data collection."

**Latency:** 2.8 seconds (LongT5-Base on CPU)

---

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/mustafasikder/academic-rag-system.git
cd academic-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Basic Usage
```python
from src.preprocessing import clean_academic_text, AcademicPaperChunker
from src.retrieval import HybridRetriever
from src.generation import RAGAnswerGenerator

# 1. Extract and clean PDF
import fitz
doc = fitz.open("paper.pdf")
text = "\n".join([page.get_text() for page in doc])
text = clean_academic_text(text)

# 2. Adaptive chunking
chunker = AcademicPaperChunker(
    target_chunk_size=600,
    min_chunk_size=200,
    max_chunk_size=1000
)
chunks = chunker.chunk_paper(text)

# 3. Index with hybrid retrieval
retriever = HybridRetriever(
    embedding_model_name='sentence-transformers/multi-qa-mpnet-base-cos-v1',
    use_cosine=True
)
retriever.index_documents(chunks)

# 4. Query and generate answer
query = "What methods were used in this study?"
retrieved_chunks, scores, indices = retriever.search(
    query, 
    k=3, 
    method='hybrid'
)

generator = RAGAnswerGenerator("google/long-t5-tglobal-base")
answer = generator.generate_answer(query, retrieved_chunks)
print(answer)
```

### Run Demo
```bash
# Web interface
python apps/gradio_app.py

# Jupyter notebook walkthrough
jupyter notebook notebooks/end_to_end_demo.ipynb
```

---

## Architecture

![diagram](/academic-rag-system/assets/architecture_diagram.png)




## Design Philosophy

### 1. Resource Constraints First
Real-world deployments often lack GPUs. System designed for:
- CPU-only inference
- Models <2GB (deployable on standard servers)

### 3. Modularity
Each component is independently swappable:
- Drop-in replacement for chunker, retriever, or generator
- Configs separate from code
- Clean interfaces between modules

---

## Roadmap

### In Progress
- [ ] Improve query-document matching (try: query expension, augmentation w embedding) 
- [ ] Unit test coverage >80%
- [ ] Expand evaluation to 50+ papers
- [ ] Docker containerization
- [ ] FastAPI backend

### Future Enhancements
- [ ] Multi-document querying (compare across papers)
- [ ] Citation extraction (link answers to specific paper sections)
- [ ] Fine-tuned retriever on academic Q&A data

---

---

## Contributing

Contributions welcome!
---


---

## Acknowledgments

- **Sentence Transformers** for embedding models
- **HuggingFace Transformers** for LLM infrastructure
- **FAISS** for efficient similarity search

---

---
