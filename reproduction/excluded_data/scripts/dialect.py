"""Blodgett, Green and O'Connor (2016) demographic dialect model ("TwitterAAE"), ported to Python 3.

Dodge et al. 2021 §5.3 use it to give each document a posterior over four dialect "topics" and assign the argmax.
The model files come from github.com/slanglab/twitteraae at a pinned commit (MIT licence) and are downloaded once.

Two inference routines:
- exact(): a line-by-line port of the repo's predict.py (collapsed variational Bayes, CVB0, sequential token
  updates, alpha=1, 5 passes; None if no token, or fewer than 20% of tokens, are in the vocabulary).
- fast(): the same model and gates, but each pass updates all tokens at once (Jacobi instead of Gauss-Seidel)
  and identical words are updated together. It is about 100x faster, which a free-tier CPU needs for millions of
  documents. scan_noblocklist.py runs both on the first documents of every shard and records how often the
  argmax agrees, so the approximation is measured, not assumed.

Assumptions (the paper does not say): the whole document is scored; tokens are lowercased runs of letters and
digits with internal apostrophes (the repo used a tweet tokenizer, twokenize, on tweets); documents the model
cannot score are reported as "unassigned".
"""
import os, re, urllib.request

import numpy as np

SHA = "db4d6d126ca26b9bb9a70d7ce18bdd5833077523"
URL = f"https://raw.githubusercontent.com/slanglab/twitteraae/{SHA}/model/"
FILES = ["model_count_table.txt", "model_vocab.txt"]
# repo order: African-American, Hispanic, Asian, White. Dodge et al. call the third "other"; the repo README notes
# that only the AA and White topics were validated.
LABELS = ["AAE", "Hispanic", "Other", "WAE"]
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")


def ensure_files(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    for f in FILES:
        p = os.path.join(model_dir, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(URL + f, p + ".tmp")
            os.replace(p + ".tmp", p)


class DialectModel:
    def __init__(self, model_dir):
        ensure_files(model_dir)
        n_wk = np.loadtxt(os.path.join(model_dir, "model_count_table.txt"))
        self.wordprobs = (n_wk + 1) / n_wk.sum(0)             # as in predict.py
        with open(os.path.join(model_dir, "model_vocab.txt"), encoding="utf-8") as f:
            vocab = [line.split("\t")[-1].strip() for line in f]
        assert len(vocab) == n_wk.shape[0]
        self.w2num = {w: i for i, w in enumerate(vocab)}
        self.K = n_wk.shape[1]

    def tokens(self, text):
        return TOKEN.findall(text.lower())

    def _gate(self, toks, thresh1, thresh2):
        ids = [self.w2num[w] for w in toks if w in self.w2num]
        if len(ids) < thresh1 or len(ids) / max(1, len(toks)) < thresh2:
            return None
        return ids

    def exact(self, toks, alpha=1, numpasses=5, thresh1=1, thresh2=0.2):
        """Port of predict.predict / infer_cvb0 (sequential updates)."""
        ids = self._gate(toks, thresh1, thresh2)
        if ids is None:
            return None
        qs = self.wordprobs[ids].copy()
        qs /= qs.sum(1, keepdims=True)
        lik = qs.copy()
        q_k = qs.sum(0)
        for _ in range(1, numpasses):
            for i in range(len(ids)):
                q_k -= qs[i]
                qs[i] = lik[i] * (q_k + alpha)
                qs[i] /= qs[i].sum()
                q_k += qs[i]
        return q_k / q_k.sum()

    def fast(self, toks, alpha=1, numpasses=5, thresh1=1, thresh2=0.2):
        """Same model and gates; all tokens updated at once per pass, identical words grouped."""
        ids = self._gate(toks, thresh1, thresh2)
        if ids is None:
            return None
        u, c = np.unique(ids, return_counts=True)
        lik = self.wordprobs[u]
        lik = lik / lik.sum(1, keepdims=True)
        qs = lik.copy()
        q_k = c @ qs
        for _ in range(1, numpasses):
            qs = lik * (q_k - qs + alpha)
            qs /= qs.sum(1, keepdims=True)
            q_k = c @ qs
        return q_k / q_k.sum()
