import re
import spacy
from utils import timeit


def tokenize_tweet(tweet_text, language_model):
    doc = language_model(tweet_text)

    token_list = []
    for token in doc:
        token_list.append(token.text)

    return token_list


def normalize_text(token_list):
    return


def remove_noise(token):
    """Removes tokens that do not add meaning or information to the data, such as punctuation, stop words, space
    characters, and hyperlinks.

    :param token: spacy.tokens.token.Token
        Token object to be checked for noise
    :return is_noise: bool
        True if the token is not noise, False otherwise
    """

    # have to cast token to str because re.match only takes in strings or byte-like objects
    matched_hyper = re.match('^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.' \
                             '[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$', str(token))
    matched_tag = re.match('(@[A-Za-z0-9_]+)', str(token))
    is_matched = bool(matched_hyper) or bool(matched_tag)

    is_noise: bool = not (token.is_punct | token.is_space | token.is_stop | is_matched)

    return is_noise


#@timeit(repeat=3, number=20)
def preprocess_tokens(docs, nlp):
    """Filter out noisy tokens and lemmatize the remaining ones.

    :param docs: list
        list of documents to be parsed
    :param nlp: spacy.lang.<code>.<language>
        spacy language, e.g. spacy.lang.es.Spanish
    :return filtered_tokens: list
        list of lists of lemmatized and filtered tokens
    """

    try:
        docs = list(docs)
    except TypeError:
        print("Input can't be casted to type 'list'")
        raise

    filtered_tokens = []
    for doc in nlp.pipe(docs):
        tokens = [token.lemma_.lower() for token in doc if (remove_noise(token) and token.lemma_ != '-PRON-')]
        filtered_tokens.append(tokens)

    return filtered_tokens
