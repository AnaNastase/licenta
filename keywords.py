import pandas as pd
import spacy
from gensim.corpora.dictionary import Dictionary
from gensim.models.phrases import Phrases
from gensim.models import LdaMulticore
import yake
import fasttext
import re
import sys
import os

STOP_WORDS = ['abstract', 'al', 'amount', 'approach', 'article', 'available', 'base', 'based', 'benefit',
              'bucharest',
              'case', 'category', 'condition', 'conference', 'context', 'copyright', 'datum', 'demonstrate',
              'demonstrates', 'demonstrated',
              'different', 'difficult', 'et', 'experiment', 'experimental', 'faculty', 'helpful', 'high',
              'ieee', 'importance', 'important', 'inconvenience', 'interest', 'interested', 'interests', 'jat',
              'jats', 'laboratory',
              'main', 'multiple', 'new', 'obtain', 'obtained', 'obtains', 'old', 'order', 'organization', 'paper',
              'people', 'policy', 'politehnica', 'polytechnic',
              'present', 'presents', 'presented', 'privacy', 'professor', 'propose', 'proposes', 'proposed',
              'quality', 'range', 'ranges', 'real',
              'recent', 'research', 'researcher', 'result', 'scale', 'show', 'shows', 'showed', 'student', 'study',
              'subject', 'studies', 'studied', 'task',
              'teacher', 'term', 'text', 'title', 'type', 'unavailable', 'university', 'useful',
              'workshop']


def clean_abstracts(abstract_list):
    """ clean up the abstract list of a researcher """
    # filter out non-English or empty abstracts
    # load fastText model
    model = fasttext.load_model('lid.176.bin')

    new_abstracts = []

    for abstract in abstract_list:
        if abstract and isinstance(abstract, str) and re.match('^(?=.*[a-zA-Z])', abstract):
            # predict the language
            predictions = model.predict(abstract)
            language = predictions[0][0].replace('__label__', '')
            # keep only texts written in English
            if language == 'en':
                new_abstracts.append(abstract)

    # remove abstracts that contain mostly person and organization names
    # load spacy model only with the ner component
    nlp = spacy.load('en_core_web_lg', exclude=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])

    clean_abstract_list = []

    for abstract in new_abstracts:
        # count the number of person and org names, and the other words separately
        person_orgs_count = 0
        other_words_count = 0

        nlp.max_length = len(abstract) + 1000
        # apply the spacy pipeline
        doc = nlp(abstract)

        for token in doc:
            if token.ent_type_ == 'PERSON' or token.ent_type_ == 'ORG':
                person_orgs_count += 1
            elif token.is_alpha:
                other_words_count += 1

        if person_orgs_count < other_words_count:
            clean_abstract_list.append(abstract)

    return clean_abstract_list


def extract_keywords_yake(abstract_list):
    """ extract keywords from a list of abstracts using YAKE """
    # concatenate the abstracts into a single string
    text = '\n'.join(abstract_list)

    # load spacy model only with the ner component
    nlp = spacy.load('en_core_web_lg', exclude=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])

    # remove some named entities with spacy
    nlp.max_length = len(text) + 1000
    doc = nlp(text)

    remove_entities = ['PERSON', 'NORP', 'FAC', 'GPE', 'LOC', 'DATE', 'TIME', 'PERCENT', 'MONEY',
                       'QUANTITY', 'CARDINAL', 'ORDINAL']

    transformed_text = ' '.join([token.text for token in doc if token.ent_type_ not in remove_entities])

    # get 15 key phrases of max 3 words
    # set parameters for yake keyword extractor
    max_ngram = 3
    deduplication_threshold = 0.5
    keywords_nr = 15
    windows_size = 1

    kw_extractor = yake.KeywordExtractor(lan="en", n=max_ngram, dedupLim=deduplication_threshold,
                                         top=keywords_nr, windowsSize=windows_size)
    # add custom stop words to the default set
    kw_extractor.stopword_set.update(set(STOP_WORDS))
    # extract keywords
    kw = kw_extractor.extract_keywords(transformed_text)

    # return a list of keywords
    keyword_list = []
    for k, _ in kw:
        keyword_list.append(k.upper())

    return keyword_list


def extract_keywords_lda(abstract_list):
    """ extract 1 topic containing 15 keywords from documents in abstract_list using LDA """
    # load spacy model
    nlp = spacy.load('en_core_web_lg')

    # keep only adjectives and nouns
    remove_pos = ['ADV', 'PRON', 'PART', 'DET', 'SPACE', 'NUM', 'SYM', 'ADP', 'VERB', 'CCONJ', 'INTJ']
    remove_entities = ['PERSON', 'NORP', 'FAC', 'GPE', 'LOC', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY',
                       'QUANTITY', 'CARDINAL', 'ORDINAL']

    # preprocess documents
    tokens = []
    for abstract in abstract_list:
        nlp.max_length = len(abstract) + 1000
        doc = nlp(abstract)
        t = [token.lemma_.upper() for token in doc if token.is_alpha and token.ent_type_ not in remove_entities
             and token.lemma_.lower() not in STOP_WORDS and token.pos_ not in remove_pos and not token.is_stop]
        tokens.append(t)

    # add n-grams to the token list
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

    # create dictionary
    dictionary = Dictionary(tokens)
    # create corpus
    corpus = [dictionary.doc2bow(text) for text in tokens]
    # apply LDA
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=1, num_topics=1, workers=3, passes=1)

    # return a list of keywords
    topic = lda_model.show_topic(topicid=-1, topn=15)
    keyword_list = [k for k, _ in topic]

    return keyword_list


if __name__ == '__main__':
    # read csv files
    authors_csv = sys.argv[1]
    publications_csv = sys.argv[2]
    authors = pd.read_csv(authors_csv)
    publications = pd.read_csv(publications_csv, sep=',')

    id_name_list = list(zip(authors['id'], authors['last_name'], authors['first_name']))
    author_names = {author_id: last_name + " " + first_name for author_id, last_name, first_name in id_name_list}

    # make a dictionary containing a list of abstracts for each author
    author_publication_pairs = list(zip(publications['user_id'], publications['abstract_text']))
    authors_texts = {author_id: [] for author_id in authors['id']}

    for author_id, abstract in author_publication_pairs:
        authors_texts[author_id].append(abstract)

    # store the results as a list of dictionaries
    results = []

    # find keywords for each author
    for author_id, abstracts in authors_texts.items():
        # clean up the abstract list
        abstract_list = clean_abstracts(authors_texts[author_id])

        print(str(author_id) + ' - ' + author_names[author_id])

        # apply YAKE
        keywords_yake = extract_keywords_yake(abstract_list)

        print('YAKE:')
        for kw in keywords_yake:
            print(kw)
        print()

        # apply LDA
        keywords_lda = extract_keywords_lda(abstract_list)

        print('LDA:')
        for kw in keywords_lda:
            print(kw)
        print()

        # add the results in the list
        keywords_yake = '; '.join(keywords_yake)
        keywords_lda = '; '.join(keywords_lda)
        results.append({'ID': author_id, 'Name': author_names[author_id],
                        'keywords YAKE': keywords_yake, 'keywords LDA': keywords_lda})

    # create data frame with pandas
    df = pd.DataFrame(results)

    # write to csv file
    out_file = 'results/' + sys.argv[3] + '.csv'
    os.makedirs('results', exist_ok=True)
    df.to_csv(out_file, index=False)
