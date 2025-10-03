import os, glob, re, json
from pathlib import Path
import faiss, numpy as np

def read_markdowns(folder="data"):
    docs = []
    for p in glob.glob(f"{folder}/*.md"):
        txt = Path(p).read_text(encoding="utf-8", errors="ignore")
        docs.append({"path": p, "text": txt})
    return docs

def chunk(text, path, size=800, overlap=120):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, buf, length = [], [], 0
    for s in sents:
        buf.append(s); length += len(s)
        if length >= size:
            chunks.append({"path": path, "text": " ".join(buf)})
            buf, length = [], 0
    if buf:
        chunks.append({"path": path, "text": " ".join(buf)})
    return chunks

def embed(texts):
    dim = 4096
    X = np.zeros((len(texts), dim), dtype="float32")
    for i,t in enumerate(texts):
        for tok in re.findall(r"[a-zA-Z0-9_]+", t.lower()):
            X[i, hash(tok) % dim] += 1.0
    X /= (np.linalg.norm(X, axis=1, keepdims=True)+1e-9)
    return X, dim

if __name__ == "__main__":
    docs = []
    for d in read_markdowns("data"):
        docs += chunk(d["text"], d["path"])
    texts = [d["text"] for d in docs]
    X, dim = embed(texts)
    idx = faiss.IndexFlatIP(dim)
    idx.add(X)
    os.makedirs("index", exist_ok=True)
    faiss.write_index(idx, "index/faiss.index")
    Path("index/meta.json").write_text(json.dumps(docs, indent=2))
    print(f"Indexed {len(docs)} chunks.")
