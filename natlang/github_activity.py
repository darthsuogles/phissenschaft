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

try:
    print('number of comments', len(comments_data))
except NameError:
    data_root = Path.home() / 'local' / 'data'
    fname = data_root / 'spark_dlp_all_comments.json'
    with open(fname) as fin:
        comments_data = json.load(fin)


df = pd.DataFrame(comments_data)

# Interact with GitHub
g = Github()
repo = g.get_repo('databricks/spark-deep-learning')
comments = []
for comment in repo.get_pulls_comments():
    comments.append(comment)

# Get only the time and date
df = pd.DataFrame(dict(time_created=s.created_at,
                       user=s.user.login,
                       text=s.body)
                  for s in comments)


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
        if tok.is_stop:
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
