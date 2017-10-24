"""
Check GitHub activity
"""

from collections import defaultdict, Counter
from pathlib import Path
import json

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
import spacy

from github import Github

import matplotlib.pyplot as plt

def fetch_original_dataset():
    # Interact with GitHub
    g = Github()
    repo = g.get_repo('databricks/spark-deep-learning')
    comments = []
    for comment in repo.get_pulls_comments():
        comments.append(comment)

    # Get only the time and date
    return pd.DataFrame(dict(time_created=s.created_at,
                             user=s.user.login,
                             text=s.body)
                        for s in comments)


try:
    print(df.columns)
except NameError:
    _DF_FNAME = "spark_deep_learning_pipelines_pr_comments.csv"
    try:
        df = pd.read_csv(_DF_FNAME)
    except:
        df = fetch_original_dataset()
        df.to_csv(_DF_FNAME)


df.groupby('user').count()
dfg = df.groupby(df.time_created.map(lambda t: t.date()))
# dfg['text'].count().plot()

""" NLP """

nlp = spacy.load('en')

# Get vocab matrix
docs = []
vocab_tok2idx = {}
vocab_idx2tok = {}
for doc in nlp.pipe(df.text):
    tok_inds = []
    for tok in doc:
        if tok.is_stop or tok.is_punct:
            continue
        text = tok.lower_
        idx = tok.lower
        tok_inds.append(idx)
        vocab_idx2tok[idx] = text
        vocab_tok2idx[text] = idx
    docs.append(tok_inds)

# Size of vocabulary
V = nlp.vocab.length

def build_sparse_matrix(docs):
    data = []
    indptr = [0]
    indices = []
    for i, wd_inds in enumerate(docs):
        wd_cnts = Counter(wd_inds)
        for idx, cnt in wd_cnts.items():
            indices.append(idx)
            data.append(cnt)
        indptr.append(len(indices))

    return csr_matrix((data, indices, indptr), dtype=int)


X = build_sparse_matrix(docs)
lda = LatentDirichletAllocation(7)
X_lda = lda.fit_transform(X)

for comp in lda.components_:
    inds = np.argsort(comp)[-10:][::-1]
    print([vocab_idx2tok[wid] for wid in inds])

for tpc_idx in range(lda.n_topics):
    print('----- TOPIC: {} -------'.format(tpc_idx))
    doc_inds = np.argsort(X_lda[:, tpc_idx])[::-1]
    docs_sel = []
    for i in doc_inds:
        doc = docs[i]
        if not doc: continue
        docs_sel.append(doc)
        print(' '.join([vocab_idx2tok[wid] for wid in doc]))
        if len(docs_sel) > 10: break


vocab_cntr = Counter()
for doc in docs:
    vocab_cntr.update(doc)
