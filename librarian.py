import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from settings.local_storage import VECTOR_DB_DIR, configure_local_storage

# 1. Setup paths
DOCS_DIR = Path("library")
DB_DIR = VECTOR_DB_DIR
DOCS_DIR.mkdir(exist_ok=True)
configure_local_storage()

# 2. Initialize the "Eyes" (Embeddings)
# We use a small model to save your 16GB RAM
embedder = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"],
)


def index_library():
    converter = DocumentConverter()
    documents = []

    print("Scanning library for new documents...")
    for pdf_path in DOCS_DIR.glob("*.pdf"):
        print(f"Parsing {pdf_path.name}...")

        # Use Docling's Rust-core to convert PDF to clean Markdown
        result = converter.convert(str(pdf_path))
        text = result.document.export_to_markdown()

        # Break into chunks so the AI doesn't get overwhelmed
        documents.append(Document(page_content=text, metadata={"source": pdf_path.name}))

    if documents:
        # Save to ChromaDB on your SSD
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embedder,
            persist_directory=str(DB_DIR),
        )
        vector_db.persist()
        print(f"Success! Indexed {len(documents)} documents to {DB_DIR}")
    else:
        print("No PDFs found in the /library folder.")


if __name__ == "__main__":
    index_library()
