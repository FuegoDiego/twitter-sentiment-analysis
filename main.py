import pandas as pd
import numpy as np
import spacy

from _process_tokenize import preprocess_tokens
from _multi_classifier_grid_search_cv import MultiClassifierGridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# load the data
print('Reading in the data....\n')
training_df = pd.read_csv('./data/tweets_training_data.csv',
                          header=None,
                          names=['target', 'id', 'date', 'flag', 'user', 'text'],
                          usecols=['target', 'id', 'date', 'text'],
                          encoding='latin-1')
print('Finished reading in the data.\n')
print('Number of rows read in:', training_df.shape[0], '\n')
training_df = training_df.groupby('target').head(50000).reset_index(drop=True)
print('Shuffling the data...\n')
training_df = training_df.sample(frac=1).reset_index(drop=True)
print('Finished shuffling the data.\n')

models = {
    'LogisticRegression': LogisticRegression()
}

# will use default parameters for MultinomialNB
model_params = {
    'LogisticRegression': {
        'penalty': ['elasticnet'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['saga'],
        'max_iter': [100],
        'n_jobs': [7],
        'l1_ratio': [0, 0.25, 0.5, 0.75, 1.0]
    }
}

transformers = {
    'CountVectorizer': CountVectorizer(),
    'TfidfTransformer': TfidfTransformer()
}

# will use default parameters for TfidfTransformer
transformer_params = {
    'CountVectorizer': {
        'min_df': np.linspace(0.005, 0.04, 3),
        'max_df': np.linspace(0.4, 0.9, 6),
        'max_features': np.linspace(5000, 75000, 5),
        'lowercase': [False],
        'ngram_range': [(1, 2), (1, 3), (1, 4)],
    },
    'TfidfTransformer': {}
}

scoring = {
    'acc': 'accuracy'
}

if __name__ == '__main__':
    docs = training_df['text']
    nlp = spacy.load('en')

    print('Tokenizing and lemmatizing...\n')
    filtered_tokens = preprocess_tokens(docs, nlp)
    print('Finished tokenizing and lemmatizing.\n')

    X = [' '.join(tokens) for tokens in filtered_tokens]
    y = training_df['target'].values

    print('Performing grid search for', len(models.keys()), 'models with', len(transformers.keys()), 'transformers each.\n')
    estimators = MultiClassifierGridSearchCV(models, model_params, transformers, transformer_params)

    estimators.fit(X, y, cv=7, scoring=scoring, n_jobs=7)
    print('Finished grid search.\n')

    score_summary = estimators.score_summary()

    print('Saving score summary...\n')
    score_summary.to_csv('./model/score_summary.csv')
    print('Score summary saved.\n')
