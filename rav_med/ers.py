"""Evidence Reliability Score (ERS) for pre-generation verification.

ERS = alpha * S - beta * C   (alpha = beta = 1 in the paper)
  S : cosine similarity between the claim/query and the passage (SBERT-style biomedical encoder)
  C : contradiction probability from an NLI model; premise = passage, hypothesis = claim

Bands: SAFE if ERS >= T; RISKY if T - 0.2 <= ERS < T; UNSAFE below that (T = 0.2 in the downstream test).
Model identifiers and revisions used for the reported results are listed in MODELS.md.
"""
from __future__ import annotations
import numpy as np

SIM_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"   # max_seq_length = 100 (the encoder default; set explicitly here)
NLI_MODEL = "roberta-large-mnli"                                        # label 0 = contradiction
SIM_MAX_LEN, NLI_MAX_LEN = 100, 256

def band(ers: float, T: float = 0.2) -> str:
    return "SAFE" if ers >= T else ("RISKY" if ers >= T - 0.2 else "UNSAFE")

def ers_from_parts(S, C, alpha: float = 1.0, beta: float = 1.0):
    return alpha * np.asarray(S, float) - beta * np.asarray(C, float)

class ERSScorer:
    """Loads the two models once. Needs `sentence-transformers` and `transformers` + `torch`."""
    def __init__(self, sim_model=SIM_MODEL, nli_model=NLI_MODEL, device=None, revisions=None):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        rev = revisions or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sim = SentenceTransformer(sim_model, device=self.device); self.sim.max_seq_length = SIM_MAX_LEN
        self.tok = AutoTokenizer.from_pretrained(nli_model, revision=rev.get(nli_model))
        self.nli = AutoModelForSequenceClassification.from_pretrained(nli_model, revision=rev.get(nli_model)).to(self.device).eval()
        self.con_idx = {v.lower(): k for k, v in self.nli.config.id2label.items()}["contradiction"]   # read from config, never hard-coded
        self._torch = torch

    def similarity(self, claims, passages):
        a = self.sim.encode(list(claims), convert_to_numpy=True, normalize_embeddings=True)
        b = self.sim.encode(list(passages), convert_to_numpy=True, normalize_embeddings=True)
        return (a * b).sum(1)

    def contradiction(self, claims, passages, batch_size=16):
        out = []
        for i in range(0, len(claims), batch_size):
            enc = self.tok(list(passages[i:i+batch_size]), list(claims[i:i+batch_size]), truncation=True,
                           max_length=NLI_MAX_LEN, padding=True, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                p = self._torch.softmax(self.nli(**enc).logits.float(), -1)[:, self.con_idx]
            out.extend(p.cpu().numpy().tolist())
        return np.array(out)

    def score(self, claims, passages, alpha=1.0, beta=1.0):
        S = self.similarity(claims, passages); C = self.contradiction(claims, passages)
        return {"S": S, "C": C, "ERS": ers_from_parts(S, C, alpha, beta)}
