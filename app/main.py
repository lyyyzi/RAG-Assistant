import os
import hashlib
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

INDEX_DIR = "data/index"

app = FastAPI(title="RAG Assistant")


# ---------- helpers ----------
def clean_snippet(text: str, limit: int = 300) -> str:
    t = " ".join(text.split())  # remove newlines + extra spaces
    if len(t) <= limit:
        return t
    cut = t[:limit]
    last_period = cut.rfind(".")
    if last_period > 80:
        return cut[: last_period + 1]
    return cut.rsplit(" ", 1)[0] + "..."


# ---------- load RAG components once ----------
embeddings = OpenAIEmbeddings()
db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 30})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------- schemas ----------
class ChatReq(BaseModel):
    question: str


# ---------- routes ----------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/files")
def list_files():
    pdfs = [f for f in os.listdir("data/raw_docs") if f.lower().endswith(".pdf")]
    return {"count": len(pdfs), "files": sorted(pdfs)}


@app.post("/chat")
def chat(req: ChatReq):
    # Retrieve docs + scores
    docs_and_scores = db.similarity_search_with_score(req.question, k=8)
    docs = [d for d, _ in docs_and_scores]

    context = "\n\n".join(
        [
            f"[{os.path.basename(d.metadata.get('source','unknown'))} p{d.metadata.get('page')}] {d.page_content}"
            for d in docs
        ]
    )

    prompt = (
        "You must answer using the provided context.\n"
        "If the question is general (like 'what is this about'), summarize the context.\n"
        "Only say \"I don't know\" if the context is empty.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.question}\n"
        "Answer:"
    )

    answer = llm.invoke(prompt).content

    sources = []
    for d, score in docs_and_scores:
        src = d.metadata.get("source", "unknown")
        filename = os.path.basename(src)

        chunk_id = hashlib.md5(
            (filename + str(d.metadata.get("page")) + d.page_content).encode("utf-8")
        ).hexdigest()[:12]

        distance = float(score)
        confidence = 1.0 / (1.0 + distance)

        sources.append(
            {
                "chunk_id": chunk_id,
                "source_file": filename,
                "source_path": src,
                "page": d.metadata.get("page"),
                "score": distance,
                "confidence": confidence,
                "snippet": clean_snippet(d.page_content, 300),
            }
        )

    return {"answer": answer, "sources": sources, "context_chars": len(context)}