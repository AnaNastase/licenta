import pandas as pd
import spacy
import pyLDAvis.gensim_models
import en_core_web_md
from gensim.corpora.dictionary import Dictionary
from gensim.models.phrases import Phrases, Phraser
from gensim.models import LdaMulticore
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import yake
import fasttext
import re

STOP_WORDS = ['abstract', 'amount', 'approach', 'article', 'available', 'base', 'based', 'benefit',
              'bucharest',
              'case', 'condition', 'conference', 'context', 'copyright', 'datum', 'demonstrate', 'demonstrates',
              'demonstrated',
              'different', 'difficult', 'experiment', 'experimental', 'faculty', 'helpful', 'high',
              'ieee', 'importance', 'important', 'inconvenience', 'interest', 'interested', 'interests', 'jat',
              'jats', 'laboratory',
              'main', 'new', 'obtain', 'obtained', 'obtains', 'old', 'order', 'organization', 'paper', 'people',
              'policy',
              'politehnica', 'polytechnic',
              'present', 'presents', 'presented', 'privacy', 'professor', 'propose', 'proposes', 'proposed',
              'quality', 'range', 'ranges', 'real',
              'recent', 'research', 'researcher', 'result', 'scale', 'show', 'shows', 'showed', 'student', 'study',
              'studies', 'studied', 'task',
              'teacher', 'term', 'text', 'title', 'type', 'unavailable', 'university', 'useful',
              'workshop']


def get_authors_texts():
    """ make a dictionary containing a list of abstracts for each author """
    author_publication_pairs = list(zip(publications['user_id'], publications['abstract_text']))

    # load fastText model
    model = fasttext.load_model('lid.176.bin')

    texts = {author_id: [] for author_id in authors["id"]}
    for author_id, abstract in author_publication_pairs:
        if abstract and isinstance(abstract, str) and re.match('^(?=.*[a-zA-Z])', abstract):
            predictions = model.predict(abstract)
            language = predictions[0][0].replace('__label__', '')
            if language == 'en':
                texts[author_id].append(abstract)

    return texts


def extract_keywords(abstract_list):
    """ extract keywords from a list of abstracts using YAKE """
    # concatenate the abstracts into a single string
    text = '\n'.join(abstract_list)

    # load spacy model
    nlp = spacy.load('en_core_web_lg')

    # entities to remove
    remove_entities = ['PERSON', 'NORP', 'FAC', 'GPE', 'LOC', 'DATE', 'TIME', 'PERCENT', 'MONEY',
                       'QUANTITY', 'CARDINAL', 'ORDINAL']

    # preprocess text
    nlp.max_length = len(text) + 1000
    doc = nlp(text)
    transformed_text = ' '.join(
        [token.text for token in doc if token.ent_type_ not in remove_entities])

    # get 15 keyphrases of max 3 words
    max_ngram = 3
    deduplication_threshold = 0.5
    keywords_nr = 15
    windowsSize = 1
    kw_extractor = yake.KeywordExtractor(lan="en", n=max_ngram, dedupLim=deduplication_threshold,
                                         top=keywords_nr, windowsSize=windowsSize)
    kw_extractor.stopword_set.update(set(STOP_WORDS))

    kw = kw_extractor.extract_keywords(transformed_text)
    return kw


def extract_topic(abstract_list):
    """ extract 1 topic containing 10 keywords from documents in abstract_list using LDA """
    # load spacy model
    nlp = spacy.load('en_core_web_lg')

    # keep only adjectives and nouns
    remove_pos = ['ADV', 'PRON', 'PART', 'DET', 'SPACE', 'NUM', 'SYM', 'ADP', 'VERB', 'CCONJ']
    remove_entities = ['PERSON', 'NORP', 'FAC', 'GPE', 'LOC', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY',
                       'QUANTITY', 'CARDINAL', 'ORDINAL']

    # preprocess documents
    tokens = []
    for abstract in abstract_list:
        nlp.max_length = len(abstract) + 1000
        doc = nlp(abstract)
        t = [token.lemma_.lower() for token in doc if token.is_alpha and token.ent_type_ not in remove_entities
             and token.lemma_.lower() not in STOP_WORDS and token.pos_ not in remove_pos and not token.is_stop]
        tokens.append(t)

    # add bigrams to the token list
    bigram = Phrases(tokens, min_count=2, delimiter=' ', threshold=1)
    tokens = [bigram[text] for text in tokens]
    trigram = Phrases(tokens, min_count=2, delimiter=' ', threshold=1)
    tokens = [trigram[text] for text in tokens]

    # remove single words (if the word count after removal is at least 100)
    ngrams = [[token for token in text if len(token.split(" ")) > 1] for text in tokens]

    word_count = 0
    for text in ngrams:
        word_count += len(text)

    if word_count > 100:
        tokens = ngrams

    # create dictionary with gensim
    dictionary = Dictionary(tokens)
    # create corpus
    corpus = [dictionary.doc2bow(text) for text in tokens]

    # apply LDA
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=300,
                             num_topics=1, workers=4, passes=50)
    return lda_model.print_topics(-1)


if __name__ == '__main__':
    # read csv files
    authors = pd.read_csv('top_20_authors.csv')
    publications = pd.read_csv('publications-top_20_authors.csv', sep=',')
    # authors = pd.read_csv('some_authors.csv')
    # publications = pd.read_csv('publications-some_authors.csv', sep=',')

    id_name_list = list(zip(authors['id'], authors['last_name'], authors['first_name']))
    author_names = {author_id: last_name + " " + first_name for author_id, last_name, first_name in id_name_list}

    # make a dictionary containing a list of abstracts for each author
    authors_texts = get_authors_texts()

    # apply YAKE and LDA to find keywords for each author
    for author_id, abstracts in authors_texts.items():
        print(str(author_id) + " - " + author_names[author_id])

        print("YAKE:")
        keywords = extract_keywords(abstracts)
        for k, _ in keywords:
            print(k)

        # print("\nLDA:")
        # topics = extract_topic(abstracts)
        # for idx, topic in topics:
        #     print(f"{topic}")

        print()
