from library_rag import LocalLibraryRAG


def main():
    rag = LocalLibraryRAG()
    added_chunks = rag.index_library()
    print(f"Indexed new chunks: {added_chunks}")


if __name__ == "__main__":
    main()
