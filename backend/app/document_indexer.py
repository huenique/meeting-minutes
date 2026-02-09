"""
Document Indexer Module for Meetily.

Handles ingestion, chunking, and embedding of local documents (markdown, PDF,
text files) into a ChromaDB vector store for retrieval-augmented generation.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Supported file extensions and their MIME types
SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".json": "application/json",
    ".rst": "text/x-rst",
    ".html": "text/html",
    ".htm": "text/html",
}


class DocumentMetadata(BaseModel):
    """Metadata about an indexed document."""
    doc_id: str
    filename: str
    file_path: str
    file_type: str
    chunk_count: int
    file_hash: str


class DocumentChunk(BaseModel):
    """Represents a chunk of a document with its embedding metadata."""
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    source_file: str


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for change detection."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _extract_text_from_file(file_path: str) -> str:
    """
    Extract text content from a supported file.

    Supports: .md, .txt, .csv, .json, .rst, .html, .htm, .pdf
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return _extract_text_from_pdf(file_path)
    elif ext in (".md", ".txt", ".csv", ".json", ".rst"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    elif ext in (".html", ".htm"):
        return _extract_text_from_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pymupdf if available, else fallback."""
    try:
        import pymupdf  # noqa: F811
        text_parts = []
        with pymupdf.open(file_path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)
    except ImportError:
        logger.warning(
            "pymupdf not installed. PDF extraction unavailable. "
            "Install with: pip install pymupdf"
        )
        return ""


def _extract_text_from_html(file_path: str) -> str:
    """Extract text from HTML by stripping tags."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    # Simple HTML tag stripping
    clean = re.sub(r"<[^>]+>", " ", content)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        overlap = chunk_size // 2

    # Try to split on paragraph boundaries first
    paragraphs = re.split(r"\n\s*\n", text)

    chunks: List[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # Keep overlap from end of current chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Single paragraph is larger than chunk_size; split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        # If a single sentence exceeds chunk_size, split by words
                        if len(sentence) > chunk_size:
                            words = sentence.split()
                            current_chunk = ""
                            for word in words:
                                if len(current_chunk) + len(word) + 1 <= chunk_size:
                                    if current_chunk:
                                        current_chunk += " " + word
                                    else:
                                        current_chunk = word
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                    current_chunk = word
                        else:
                            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class DocumentIndexer:
    """
    Manages document ingestion and indexing using ChromaDB for vector storage.

    Uses ChromaDB's built-in embedding function (all-MiniLM-L6-v2) by default.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "meetily_knowledge_base",
    ):
        """
        Initialize the document indexer.

        Args:
            persist_directory: Directory for ChromaDB persistence.
                Defaults to 'knowledge_base' in the backend directory.
            collection_name: Name of the ChromaDB collection.
        """
        self._persist_directory = persist_directory or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "knowledge_base"
        )
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        logger.info(
            f"DocumentIndexer initialized (persist_dir={self._persist_directory})"
        )

    def _get_collection(self):
        """Lazily initialize and return the ChromaDB collection."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb

            self._client = chromadb.PersistentClient(
                path=self._persist_directory
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"description": "Meetily knowledge base for RAG"},
            )
            logger.info(
                f"ChromaDB collection '{self._collection_name}' ready "
                f"with {self._collection.count()} documents"
            )
            return self._collection
        except ImportError:
            raise ImportError(
                "chromadb is required for document indexing. "
                "Install with: pip install chromadb"
            )

    def index_file(
        self,
        file_path: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> DocumentMetadata:
        """
        Index a single file into the knowledge base.

        Args:
            file_path: Path to the file to index.
            chunk_size: Maximum characters per chunk.
            overlap: Overlap between consecutive chunks.

        Returns:
            DocumentMetadata with indexing results.
        """
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
            )

        file_hash = _compute_file_hash(file_path)
        doc_id = hashlib.sha256(file_path.encode()).hexdigest()[:16]

        collection = self._get_collection()

        # Check if document already indexed with same hash
        existing = collection.get(
            where={"doc_id": doc_id},
        )
        if existing and existing["ids"]:
            # Check if hash changed
            existing_hashes = [
                m.get("file_hash") for m in existing["metadatas"] if m
            ]
            if file_hash in existing_hashes:
                logger.info(f"File already indexed and unchanged: {file_path}")
                return DocumentMetadata(
                    doc_id=doc_id,
                    filename=os.path.basename(file_path),
                    file_path=file_path,
                    file_type=ext,
                    chunk_count=len(existing["ids"]),
                    file_hash=file_hash,
                )
            # Remove old chunks for this document
            collection.delete(ids=existing["ids"])
            logger.info(f"Removed {len(existing['ids'])} old chunks for: {file_path}")

        # Extract and chunk text
        text = _extract_text_from_file(file_path)
        if not text.strip():
            logger.warning(f"No text extracted from: {file_path}")
            return DocumentMetadata(
                doc_id=doc_id,
                filename=os.path.basename(file_path),
                file_path=file_path,
                file_type=ext,
                chunk_count=0,
                file_hash=file_hash,
            )

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return DocumentMetadata(
                doc_id=doc_id,
                filename=os.path.basename(file_path),
                file_path=file_path,
                file_type=ext,
                chunk_count=0,
                file_hash=file_hash,
            )

        # Add chunks to ChromaDB
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": os.path.basename(file_path),
                "file_path": file_path,
                "file_type": ext,
                "chunk_index": i,
                "file_hash": file_hash,
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=metadatas,
        )

        logger.info(
            f"Indexed {len(chunks)} chunks from: {os.path.basename(file_path)}"
        )
        return DocumentMetadata(
            doc_id=doc_id,
            filename=os.path.basename(file_path),
            file_path=file_path,
            file_type=ext,
            chunk_count=len(chunks),
            file_hash=file_hash,
        )

    def index_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[DocumentMetadata]:
        """
        Index all supported files in a directory.

        Args:
            directory_path: Path to the directory to scan.
            recursive: Whether to scan subdirectories.
            chunk_size: Maximum characters per chunk.
            overlap: Overlap between consecutive chunks.

        Returns:
            List of DocumentMetadata for all indexed files.
        """
        directory_path = os.path.abspath(directory_path)
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        results: List[DocumentMetadata] = []
        pattern = "**/*" if recursive else "*"

        for ext in SUPPORTED_EXTENSIONS:
            for file_path in Path(directory_path).glob(f"{pattern}{ext}"):
                try:
                    metadata = self.index_file(
                        str(file_path),
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                    results.append(metadata)
                except Exception as e:
                    logger.error(f"Error indexing {file_path}: {e}")

        logger.info(
            f"Indexed {len(results)} files from directory: {directory_path}"
        )
        return results

    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict]:
        """
        Search the knowledge base for relevant chunks.

        Args:
            query: The search query text.
            n_results: Maximum number of results to return.

        Returns:
            List of dicts with 'text', 'metadata', and 'distance' keys.
        """
        collection = self._get_collection()

        if collection.count() == 0:
            logger.info("Knowledge base is empty, no results to return")
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )

        search_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                search_results.append(
                    {
                        "text": doc,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "distance": results["distances"][0][i]
                        if results["distances"]
                        else None,
                    }
                )

        logger.info(
            f"Search for '{query[:50]}...' returned {len(search_results)} results"
        )
        return search_results

    def list_documents(self) -> List[Dict]:
        """List all indexed documents (deduplicated by doc_id)."""
        collection = self._get_collection()
        all_data = collection.get()

        docs: Dict[str, Dict] = {}
        if all_data and all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                doc_id = meta.get("doc_id", "")
                if doc_id and doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "filename": meta.get("filename", ""),
                        "file_path": meta.get("file_path", ""),
                        "file_type": meta.get("file_type", ""),
                        "file_hash": meta.get("file_hash", ""),
                    }

        return list(docs.values())

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document and all its chunks from the knowledge base.

        Args:
            doc_id: The document ID to remove.

        Returns:
            True if the document was found and removed.
        """
        collection = self._get_collection()
        existing = collection.get(where={"doc_id": doc_id})

        if not existing or not existing["ids"]:
            return False

        collection.delete(ids=existing["ids"])
        logger.info(
            f"Removed document {doc_id} ({len(existing['ids'])} chunks)"
        )
        return True

    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        collection = self._get_collection()
        docs = self.list_documents()
        return {
            "total_chunks": collection.count(),
            "total_documents": len(docs),
            "collection_name": self._collection_name,
            "persist_directory": self._persist_directory,
        }
