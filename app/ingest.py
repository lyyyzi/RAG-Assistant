import os
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

RAW_DIR = "data/raw_docs"
INDEX_DIR = "data/index"

def load_pdfs():
    docs = []
    for fn in os.listdir(RAW_DIR):
        if fn.endswith(".pdf"):
            path = os.path.join(RAW_DIR, fn)
            docs.extend(PyPDFLoader(path).load())
    return docs
print("Starting ingestion...")
def main():
    docs = load_pdfs()
    print(f"Loaded {len(docs)} documents")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(INDEX_DIR)

    print("Index built successfully!")

if __name__ == "__main__":
    main()