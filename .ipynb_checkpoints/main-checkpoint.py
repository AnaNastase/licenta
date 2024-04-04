import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import spacy
import pyLDAvis.gensim_models
import en_core_web_md
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel

# read csv
authors = pd.read_csv('top_20_authors.csv')

publications = pd.read_csv('publications-top_20_authors.csv', sep=',')
print(publications['abstract_text'][0])

# tokenize, lemmatize, remove stop words
nlp = spacy.load('en_core_web_md')
remove_pos = ['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM']

tokens = []
for abstract in publications['abstract_text']:
    if abstract and isinstance(abstract, str):
        abstract_nlp = nlp(abstract)

        t = []
        for tok in abstract_nlp:
            if tok.pos_ not in remove_pos and not tok.is_stop and tok.is_alpha:
                t.append(tok.lemma_.lower())

        tokens.append(t)

# create dictionary with gensim
dictionary = Dictionary(tokens)
print(dictionary.token2id)
