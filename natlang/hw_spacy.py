"""
Using spaCy for natural language processing
"""

from pathlib import Path
import spacy
from spacy import displacy
from spacy.matcher import PhraseMatcher
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

text = """\
We concur with European Directors regarding the importance of
continued focus on fiscal consolidation. We note the medium term targets
discussed in the buff statement, but are conscious of risks to future budget
deficit targets, and view options for further tax cutting or spending cuts as
becoming limited. What probability does the staff attach to the risk that the
budget deficit targets may not be met?
"""

def init_spacy(lang_model='en_core_web_lg'):
    from spacy.lang.en.stop_words import STOP_WORDS
    nlp = spacy.load(lang_model)
    # Missing stop words in a few models
    # https://github.com/explosion/spaCy/issues/922
    nlp.vocab.add_flag(
        lambda s: s.lower() in STOP_WORDS, spacy.attrs.IS_STOP)
    return nlp

# REF: https://spacy.io/usage/#section-troubleshooting
try: print(nlp.pipe_names)
# https://spacy.io/models/en#section-en_core_web_lg
# This model also contains GloVe vectors (embeded in a 300 dimesional vector space).
except: nlp = init_spacy()

# The default model gives us the basic tokenization and dependency parsing.
# REF: https://spacy.io/api/#section-nn-model
doc = nlp(text)

# TODO: add custom phase matching
def get_phrase_matcher(nlp, terminologies):
    # With the rich linguistic annotations, spaCy allows us to
    # match phrases in a very flexible manner.
    # https://spacy.io/usage/linguistic-features#section-rule-based-matching
    matcher = PhraseMatcher(nlp.vocab)
    _patterns = [nlp(phrase) for phrase in terminologies]
    matcher.add('TerminologyList', None, *_patterns)
    return matcher

# Get a matcher for a set of special phrases in our problem domain
matcher = get_phrase_matcher(nlp, [
    'European Directors',
    'IMF', 'International Monetary Funds',
    'European Union',
])

for _, init, bnd in matcher(doc):
    print(doc[init:bnd])

for i, sent in enumerate(doc.sents):
    print('sentence {}\t: {} [{}]'.format(i, sent, sent.sentiment))

for tok in sent:
    if tok.is_stop or tok.is_punct or tok.is_space: continue
    print(tok)

# displacy.serve(doc, style='dep')

# To find a summary on the corpus, LDA is a good idea.
# https://radimrehurek.com/gensim/models/ldamulticore.html

# We can also summarize the article via TensorFlow's TextSum
# https://github.com/tensorflow/models/tree/master/research/textsum

# Collect the word vectors and plot them
def plot_tsne(X):
    X_embedded = TSNE(n_components=2).fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1])
    fig.show()

# More dataset
fp_data_root = Path.home() / 'local' / 'data'
# TODO: load these datasets and inspect
fp_news_corpus = fp_data_root / 'econ_poly_corpus'

# REF: http://blog.kaggle.com/2015/07/27/crowdflower-winners-interview-1st-place-chenglong-chen/
# REF: https://www.kaggle.com/kaggle/hillary-clinton-emails
# REF: http://www.casmacat.eu/corpus/news-commentary.html
# REF: http://www.statmt.org/europarl/
# Must specify the encoding correctly!
df_us_econ_news = pd.read_csv(
    fp_news_corpus / 'Full-Economic-News-DFE-839861.csv',
    encoding="latin")
    #encoding="iso-8859-1")

# TODO: check why elpy is so slow
X_orig_ = []
corpus_inds = []
corpus = {}
for el in df_us_econ_news.itertuples():
    doc_idx = el.articleid
    doc = nlp(el.text)
    for sent_idx, sent in enumerate(doc.sents):
        idx = (doc_idx, sent_idx)
        corpus[idx] = sent
        corpus_inds.append(idx)
        X_orig_.append(sent.vector)

# Build the t-SNE 2D vectors
X_w2v = np.vstack(X_orig_); del X_orig_

# X_embedded = TSNE(n_components=2).fit_transform(X_w2v)

# for vec2, idx in zip(X_embeded, corpus_inds):
#     {'x': vec2[0],
#      'y': vec2[1],
#      'text': corpus[idx].text}
