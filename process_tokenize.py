def tokenize_tweet(tweet_text, language_model):
    doc = language_model(tweet_text)

    token_list = []
    for token in doc:
        token_list.append(token.text)

    return token_list


def normalize_text(token_list):
    return


def remove_noise(token_list, stop_words):
    return


def preprocess_tweets(tweet_text):
    return
