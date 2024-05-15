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
from langdetect import detect, detect_langs, DetectorFactory
import re


def get_authors_texts():
    """ make a dictionary containing a list of abstracts for each author """
    author_publication_pairs = list(zip(publications['user_id'], publications['abstract_text']))

    authors_texts = {author_id: [] for author_id in authors["id"]}
    for author_id, abstract in author_publication_pairs:
        if abstract and isinstance(abstract, str) and re.match('^(?=.*[a-zA-Z])', abstract):
            language = detect(abstract)
            if language == 'en':
                authors_texts[author_id].append(abstract)

    return authors_texts


def extract_keywords(author_names):
    """ extract topics for each author using yake """
    # make a dictionary containing the combined abstracts for each author
    authors_texts = {}
    for author_id, texts in get_authors_texts().items():
        authors_texts[author_id] = '\n'.join(texts)

    # load spacy model
    nlp = spacy.load('en_core_web_md')

    # entities/words to remove
    remove_entities = ['PERSON', 'NORP', 'FAC', 'GPE', 'LOC', 'DATE', 'TIME', 'PERCENT', 'MONEY',
                       'QUANTITY', 'CARDINAL', 'ORDINAL']
    stop_words = ['paper', 'present', 'propose', 'show', 'datum', 'people', 'result', 'solution', 'case', 'order',
                  'base', 'ieee', 'privacy', 'policy', 'new', 'old', 'context', 'high', 'different', 'research', 'type',
                  'approach', 'important', 'main', 'range', 'helpful', 'large', 'difficult', 'available', 'amount',
                  'useful', 'importance', 'article', 'abstract', 'scale', 'copyright', 'real', 'quality', 'demonstrate',
                  'inconvenience', 'benefit', 'unavailable', 'term', 'condition', 'interest', 'recent', 'obtain',
                  'title', 'jat', 'jats',
                  'organization', 'task', 'student', 'professor', 'teacher', 'university', 'workshop', 'study', 'text',
                  'conference']

    for author_id, text in authors_texts.items():
        # preprocess text
        doc = nlp(text)
        transformed_text = ' '.join(
            [token.text for token in doc if token.ent_type_ not in remove_entities
             and token.lemma_.lower() not in stop_words])

        print(str(author_id) + " - " + author_names[author_id])

        # get 15 keyphrases of max 3 words
        max_ngram = 3
        deduplication_threshold = 0.5
        keywords_nr = 15
        custom_kw_extractor = yake.KeywordExtractor(lan="en", n=max_ngram, dedupLim=deduplication_threshold,
                                                    top=keywords_nr, features=None)

        keywords = custom_kw_extractor.extract_keywords(transformed_text)
        for kw in keywords:
            print(kw)

        print()


if __name__ == '__main__':
    # read csv files
    authors = pd.read_csv('top_20_authors.csv')
    publications = pd.read_csv('publications-top_20_authors.csv', sep=',')

    id_name_list = list(zip(authors['id'], authors['last_name'], authors['first_name']))
    author_names = {author_id: last_name + " " + first_name for author_id, last_name, first_name in id_name_list}

    extract_keywords(author_names)
