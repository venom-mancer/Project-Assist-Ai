from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LocalLibraryRAG:
    """Indexes local PDFs and retrieves relevant chunks for a query."""

    def __init__(
        self,
        library_dir="library",
        persist_dir="library/.chroma",
        embedding_model="nomic-embed-text",
    ):
        self.library_dir = Path(library_dir)
        self.persist_dir = str(Path(persist_dir))
        self.converter = DocumentConverter()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=180,
        )
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.db = Chroma(
            collection_name="assist_eye_library",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        self._is_indexed = False

    def _pdf_paths(self):
        return sorted(self.library_dir.glob("*.pdf"))

    def _indexed_sources(self):
        docs = self.db.get(include=["metadatas"])
        metadatas = docs.get("metadatas", [])
        return {meta.get("source") for meta in metadatas if meta and meta.get("source")}

    def _docling_to_markdown(self, pdf_path):
        result = self.converter.convert(str(pdf_path))
        return result.document.export_to_markdown()

    def index_library(self, force_reindex=False):
        pdf_paths = self._pdf_paths()
        if not pdf_paths:
            self._is_indexed = True
            return 0

        indexed_sources = set() if force_reindex else self._indexed_sources()
        new_chunks = []

        for pdf_path in pdf_paths:
            source = str(pdf_path)
            if source in indexed_sources:
                continue

            markdown = self._docling_to_markdown(pdf_path)
            chunks = self.splitter.split_text(markdown)
            new_chunks.extend(chunks)
            self.db.add_texts(
                texts=chunks,
                metadatas=[{"source": source}] * len(chunks),
            )

        # Explicit SSD persistence to reduce RAM pressure on future runs.
        if new_chunks:
            self.db.persist()

        self._is_indexed = True
        return len(new_chunks)

    def retrieve_context(self, query, k=4):
        if not self._is_indexed:
            self.index_library()

        docs = self.db.similarity_search(query, k=k)
        if not docs:
            return "No local library matches found."

        formatted_docs = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            formatted_docs.append(f"[{i}] Source: {source}\n{doc.page_content}")
        return "\n\n".join(formatted_docs)
