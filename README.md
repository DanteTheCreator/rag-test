# RAG Q&A Assistant

A command-line Question-Answering assistant built with **LlamaIndex** and **Ollama** that uses Retrieval-Augmented Generation (RAG) to answer questions based on your document corpus.

---

## 🚀 Features

- **Local LLM**: Uses Ollama for completely local operation
- **RAG Pipeline**: Implements complete ingestion → embedding → retrieval → generation pipeline
- **Smart Caching**: Automatically caches processed documents for faster subsequent queries
- **Multiple Formats**: Supports Markdown, plain text, Python, JSON, YAML, and more
- **Flexible Configuration**: Customizable models, chunk sizes, and retrieval parameters

---

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
3. Required models pulled in Ollama:
   ```bash
   ollama pull gemma3:4b  # or your preferred LLM
   ollama pull nomic-embed-text  # for embeddings
   ```
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🗂️ Project Structure

- `qa.py` — Main CLI script implementing the RAG pipeline
- `docs/` — Place your Markdown, text, or code documentation here
- `cache/` — Stores vector index and cache files for fast retrieval
- `requirements.txt` — Python dependencies

---

## ⚙️ Usage

Ask a question from the command line:

```bash
python qa.py --question "How do I reset my password?"
```

**Options:**

- `--question` — Your question (required)
- `--docs-dir` — Directory containing documents (default: `./docs`)
- `--cache-dir` — Directory for caching/index (default: `./cache`)
- `--ollama-host` — Ollama server URL (default: `http://localhost:11434`)
- `--model-name` — LLM model name (default: `gemma3:4b`)
- `--embed-model` — Embedding model name (default: `nomic-embed-text`)
- `--top-k` — Number of relevant chunks to retrieve (default: 3)
- `--refresh` — Force re-indexing of documents
- `--info` — Show corpus statistics
- `--verbose` — Enable debug logging

**Examples:**

```bash
python qa.py --question "How do I reset my password?"
python qa.py --question "Explain the authentication process" --top-k 5
python qa.py --info
```

---

## 🧠 How It Works

The pipeline consists of four main stages:

1. **Ingestion**: Reads and chunks all files in `./docs/` (supports `.md`, `.txt`, `.py`, `.json`, `.yaml`, `.yml`, `.rst`, `.html`)
2. **Embedding & Indexing**: Converts chunks to embeddings using Ollama, builds a vector store, and caches it in `./cache/storage/`
3. **Retrieval**: Given a user query, fetches the top-k most relevant chunks
4. **Generation**: Combines retrieved chunks and the user query, then uses the LLM to generate a concise answer

---

## 🛠️ Code Navigation

- **`qa.py`**: Main script. Contains the `RAGQAAssistant` class and CLI interface. Key methods:
  - `ingest_documents()`: Loads and chunks documents
  - `embed_and_index()`: Embeds and indexes documents
  - `generate_answer()`: Retrieves and generates answers
  - `process_question()`: Orchestrates the full RAG pipeline
- **`docs/`**: Place your documentation files here
- **`cache/`**: Stores the vector index and cache files

---
