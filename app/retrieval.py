import json, re
from pathlib import Path
import numpy as np, faiss

def _embed_query(q, dim):
    x = np.zeros((dim,), dtype="float32")
    for tok in re.findall(r"[a-zA-Z0-9_]+", q.lower()):
        x[hash(tok) % dim] += 1.0
    x /= (np.linalg.norm(x)+1e-9)
    return x.reshape(1, -1)

def search(q, k=5, index_dir="index"):
    idx = faiss.read_index(f"{index_dir}/faiss.index")
    meta = json.loads(Path(f"{index_dir}/meta.json").read_text())
    qv = _embed_query(q, idx.d)
    sims, I = idx.search(qv, k)
    results=[]
    for rank,i in enumerate(I[0]):
        if i<0: continue
        results.append({"rank":rank+1,"score":float(sims[0][rank]),**meta[i]})
    return results
