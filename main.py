import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import spacy
import pyLDAvis.gensim_models
import en_core_web_md
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore, Phrases, LdaModel
from gensim.models import CoherenceModel

# read csv
authors = pd.read_csv('top_20_authors.csv')
publications = pd.read_csv('publications-top_20_authors.csv', sep=',')

authors_names = list(zip(authors["last_name"], authors["first_name"], authors["id"]))
authors_names = [(name[0].upper(), name[1].split(" ")[0].upper(), name[2]) for name in list(authors_names)]

authors_texts = {}
all_texts = list(zip(publications['abstract_text'], publications['authors']))

# find each author's publications
for last_name, first_name, author_id in authors_names:
    abstracts = []
    for abstract, authors in all_texts:
        if abstract and isinstance(abstract, str) and authors and isinstance(authors, str):
            # check if the current author is one of the authors of this publication
            authors = authors.split(";")
            for a in authors:
                a = a.upper()
                if last_name in a and (first_name in a or (first_name[0] + ".") in a):
                    abstracts.append(abstract)
                    break
    authors_texts[author_id] = abstracts

# get one topic for each author
authors_topics = {}
nlp = spacy.load('en_core_web_md')
remove_pos = ['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM']

for author_id in authors_texts:
    texts = authors_texts[author_id]

    # tokenize, lemmatize, remove stop words
    tokens = []
    for abstract in texts:
        if abstract and isinstance(abstract, str):
            abstract_nlp = nlp(abstract)

            t = []
            for tok in abstract_nlp:
                if tok.pos_ not in remove_pos and not tok.is_stop and tok.is_alpha:
                    t.append(tok.lemma_.lower())

            tokens.append(t)

    print(author_id)
    print()

    # add bigrams to the token list
    # bigram = Phrases(tokens)
    # tokens = [bigram[text] for text in tokens]
    #
    # for i, abstract in enumerate(tokens):
    #     for j, token in enumerate(abstract):
    #         tokens[i][j] = token.replace("_", " ")

    # create dictionary with gensim
    dictionary = Dictionary(tokens)

    # create corpus
    corpus = [dictionary.doc2bow(text) for text in tokens]

    # apply lda
    # lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=50, num_topics=1, workers=4, passes=10)
    # lda_model.print_topics(-1)
    lda_model = LdaModel(corpus=corpus, num_topics=1, id2word=dictionary)
    lda_model.show_topics()
    print()
