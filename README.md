# RAG Q&A Assistant

A command-line Question-Answering assistant built with **LlamaIndex** and **Ollama** that uses Retrieval-Augmented Generation (RAG) to answer questions based on your document corpus.

## 🚀 Features

- **Local LLM**: Uses Ollama for completely local operation
- **RAG Pipeline**: Implements complete ingestion → embedding → retrieval → generation pipeline
- **Smart Caching**: Automatically caches processed documents for faster subsequent queries
- **Multiple Formats**: Supports Markdown, plain text, Python, JSON, YAML, and more
- **Flexible Configuration**: Customizable models, chunk sizes, and retrieval parameters

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
3. Required models pulled in Ollama:
   ```bash
   ollama pull gemma3:4b  # or your preferred LLM
   ollama pull nomic-embed-text  # for embeddings