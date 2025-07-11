#!/usr/bin/env python3
"""
RAG-based Q&A Assistant using LlamaIndex and Ollama
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

try:
    # Updated imports for newer LlamaIndex versions
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings 
    from llama_index.core.node_parser import SimpleNodeParser 
    from llama_index.core.storage.storage_context import StorageContext 
    from llama_index.core.storage.docstore import SimpleDocumentStore   
    from llama_index.core.storage.index_store import SimpleIndexStore   
    from llama_index.core.vector_stores import SimpleVectorStore 
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama 
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔄 Trying alternative import paths...")
    
    try:
        # Fallback imports for older versions
        from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document, ServiceContext
        from llama_index.node_parser import SimpleNodeParser
        from llama_index.storage.storage_context import StorageContext
        from llama_index.storage.docstore import SimpleDocumentStore
        from llama_index.storage.index_store import SimpleIndexStore
        from llama_index.vector_stores import SimpleVectorStore
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        
        # For older versions, we need to use ServiceContext instead of Settings
        USE_SERVICE_CONTEXT = True
    except ImportError as e2:
        print(f"❌ Failed to import required packages: {e2}")
        print("📦 Please install with: pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama")
        sys.exit(1)
else:
    USE_SERVICE_CONTEXT = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGQAAssistant:
    """RAG-based Question-Answering Assistant"""
    
    def __init__(
        self,
        docs_dir: str = "./docs",
        cache_dir: str = "./cache",
        ollama_host: str = "http://localhost:11434",
        model_name: str = "gemma3:4b",
        embed_model: str = "nomic-embed-text",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize the RAG Q&A Assistant"""
        self.docs_dir = Path(docs_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.storage_dir = self.cache_dir / "storage"
        self.storage_dir.mkdir(exist_ok=True)
        
        # Validate documents directory
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"📁 Documents directory '{docs_dir}' does not exist")
        
        # Initialize Ollama components
        self._initialize_ollama(ollama_host, model_name, embed_model)
        
        # Initialize text processing
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        self.index = None
        self.query_engine = None
        
        # Supported file extensions
        self.supported_extensions = {'.txt', '.md', '.py', '.json', '.yaml', '.yml', '.rst', '.html'}

    def _initialize_ollama(self, ollama_host: str, model_name: str, embed_model: str):
        """Initialize Ollama LLM and embedding models"""
        try:
            logger.info(f"🔧 Initializing Ollama with model: {model_name}")
            self.llm = Ollama(
                model=model_name, 
                base_url=ollama_host, 
                request_timeout=120.0
            )
            
            logger.info(f"🔧 Initializing embeddings with model: {embed_model}")
            self.embed_model = OllamaEmbedding(
                model_name=embed_model, 
                base_url=ollama_host
            )
            
            # Configure global settings based on version
            if not USE_SERVICE_CONTEXT:
                # Newer versions use Settings
                Settings.llm = self.llm
                Settings.embed_model = self.embed_model
            else:
                # Older versions use ServiceContext
                from llama_index import ServiceContext
                self.service_context = ServiceContext.from_defaults(
                    llm=self.llm,
                    embed_model=self.embed_model,
                    node_parser=self.node_parser
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama: {e}")
            raise

    def ingest_documents(self) -> List[Document]:
        """Stage 1: Ingestion - Read all files from docs directory"""
        logger.info("📖 Stage 1: Ingesting documents...")
        
        documents = []
        files_processed = 0
        
        for file_path in self.docs_dir.rglob("*"):
            if not file_path.is_file():
                continue
                
            if file_path.suffix.lower() not in self.supported_extensions:
                logger.debug(f"⏭️  Skipping unsupported file: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    logger.warning(f"⚠️  Empty file: {file_path}")
                    continue
                
                # Create document with metadata
                doc = Document(
                    text=content,
                    metadata={
                        "filename": file_path.name,
                        "file_path": str(file_path.relative_to(self.docs_dir)),
                        "file_type": file_path.suffix.lower(),
                        "size": len(content)
                    }
                )
                documents.append(doc)
                files_processed += 1
                logger.debug(f"✅ Loaded: {file_path.name}")
                
            except Exception as e:
                logger.warning(f"❌ Error loading {file_path}: {e}")
                continue
        
        if not documents:
            raise ValueError(f"❌ No documents found in {self.docs_dir}")
        
        logger.info(f"✅ Ingested {len(documents)} documents from {files_processed} files")
        return documents

    def embed_and_index(self, documents: List[Document], force_refresh: bool = False):
        """Stage 2: Embed & Index - Convert documents to embeddings and create vector store"""
        # Try to load existing index first
        if not force_refresh and self._load_existing_index():
            logger.info("✅ Using existing index")
            return
        
        logger.info("🔄 Stage 2: Embedding and indexing documents...")
        
        try:
            # Create storage context
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                vector_store=SimpleVectorStore(),
                index_store=SimpleIndexStore(),
            )
            
            # Create vector store index based on version
            logger.info("🔧 Creating vector store index...")
            if not USE_SERVICE_CONTEXT:
                # Newer version
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=True
                )
            else:
                # Older version
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    service_context=self.service_context,
                    show_progress=True
                )
            
            # Save index for future use
            self._save_index()
            
            logger.info("✅ Embedding and indexing completed")
            
        except Exception as e:
            logger.error(f"❌ Error in embedding/indexing: {e}")
            raise

    def generate_answer(self, query: str, top_k: int = 3) -> str:
        """Stage 4: Generate - Generate answer using retrieved chunks and LLM"""
        logger.info("🤖 Stage 4: Generating answer...")
        
        if not self.index:
            raise RuntimeError("❌ Index not available. Please process documents first.")
        
        try:
            # Create query engine if not exists
            if not self.query_engine:
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=top_k,
                    response_mode="compact"
                )
            
            # Generate response
            response = self.query_engine.query(query)
            answer = str(response)
            
            logger.info("✅ Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error in answer generation: {e}")
            raise

    def process_question(self, question: str, top_k: int = 3, force_refresh: bool = False) -> str:
        """Complete RAG pipeline: Ingest -> Embed -> Retrieve -> Generate"""
        if not question.strip():
            raise ValueError("❌ Question cannot be empty")
        
        logger.info(f"❓ Processing question: {question}")
        
        try:
            # Stage 1: Ingestion (only if index doesn't exist or force refresh)
            if not self.index or force_refresh:
                documents = self.ingest_documents()
                # Stage 2: Embed & Index
                self.embed_and_index(documents, force_refresh)
            
            # Stage 3 & 4: Retrieve & Generate (combined in query engine)
            answer = self.generate_answer(question, top_k)
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            raise

    def _save_index(self):
        """Save the index to disk"""
        try:
            if self.index:
                self.index.storage_context.persist(persist_dir=str(self.storage_dir))
                logger.info(f"💾 Index saved to {self.storage_dir}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save index: {e}")

    def _load_existing_index(self) -> bool:
        """Load existing index from disk"""
        try:
            if (self.storage_dir / "docstore.json").exists():
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.storage_dir)
                )
                
                if not USE_SERVICE_CONTEXT:
                    # Newer version
                    self.index = VectorStoreIndex.from_vector_store(
                        storage_context.vector_store,
                        storage_context=storage_context
                    )
                else:
                    # Older version
                    self.index = VectorStoreIndex.from_vector_store(
                        storage_context.vector_store,
                        storage_context=storage_context,
                        service_context=self.service_context
                    )
                
                logger.info(f"📂 Loaded existing index from {self.storage_dir}")
                return True
        except Exception as e:
            logger.warning(f"⚠️  Failed to load existing index: {e}")
        return False

    def get_corpus_info(self) -> dict:
        """Get information about the document corpus"""
        if not self.docs_dir.exists():
            return {"error": "Documents directory not found"}
        
        file_count = 0
        total_size = 0
        file_types = {}
        
        for file_path in self.docs_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                file_count += 1
                size = file_path.stat().st_size
                total_size += size
                ext = file_path.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            "total_files": file_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": file_types,
            "docs_directory": str(self.docs_dir)
        }


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='RAG-based Q&A Assistant using LlamaIndex and Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qa.py --question "How do I reset my password?"
  python qa.py --question "What is the main topic?" --top-k 5
  python qa.py --question "Explain the process" --refresh
  python qa.py --info  # Show corpus information
        """
    )
    
    parser.add_argument('--question', type=str, help='The question to answer')
    parser.add_argument('--docs-dir', type=str, default='./docs', help='Directory containing documents')
    parser.add_argument('--cache-dir', type=str, default='./cache', help='Directory for caching')
    parser.add_argument('--ollama-host', type=str, default='http://localhost:11434', help='Ollama server URL')
    parser.add_argument('--model-name', type=str, default='gemma3:4b', help='Ollama model name')
    parser.add_argument('--embed-model', type=str, default='nomic-embed-text', help='Embedding model name')
    parser.add_argument('--top-k', type=int, default=3, help='Number of relevant chunks to retrieve')
    parser.add_argument('--refresh', action='store_true', help='Force refresh of document index')
    parser.add_argument('--info', action='store_true', help='Show corpus information')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
    
    try:
        # Initialize the assistant
        assistant = RAGQAAssistant(
            docs_dir=args.docs_dir,
            cache_dir=args.cache_dir,
            ollama_host=args.ollama_host,
            model_name=args.model_name,
            embed_model=args.embed_model
        )
        
        # Show corpus information
        if args.info:
            info = assistant.get_corpus_info()
            print("\n📊 Document Corpus Information:")
            print(f"📁 Directory: {info['docs_directory']}")
            print(f"📄 Total files: {info['total_files']}")
            print(f"💾 Total size: {info['total_size_mb']} MB")
            print(f"🗂️  File types: {info['file_types']}")
            return
        
        # Process question
        if not args.question:
            print("❌ Error: Please provide a question using --question")
            print("💡 Use --help for more information")
            sys.exit(1)
        
        print(f"\n❓ Question: {args.question}")
        print("⏳ Processing...")
        
        answer = assistant.process_question(
            args.question, 
            top_k=args.top_k, 
            force_refresh=args.refresh
        )
        
        print(f"\n🤖 Answer: {answer}")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()