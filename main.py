import pandas as pd
import spacy

from process_tokenize import preprocess_tokens
from utils import MultiClassifierGridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
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
    'MultinomialNB': MultinomialNB(),
    'SGDClassifier': SGDClassifier(),
    'LogisticRegression': LogisticRegression()
}

# will use default parameters for MultinomialNB
model_params = {
    'SGDClassifier': {
        'penalty': ['elasticnet'],
        'alpha': [0.0001, 0.0005, 0.001, 0.01],
        'l1_ratio': [0, 0.25, 0.5, 0.75, 1.0],
        'max_iter': [1000, 2500, 5000, 10000],
        'n_jobs': [7],
        'learning_rate': ['optimal', 'adaptive'],
        'eta0': [0.01],
        'early_stopping': [True]
    },
    'LogisticRegression': {
        'penalty': ['elasticnet'],
        'C': [0.5, 1.0],
        'solver': ['saga'],
        'max_iter': [100, 500, 1000, 5000],
        'n_jobs': [7],
        'l1_ratio': [0, 0.25, 0.5, 0.75, 1.0]
    },
    'MultinomialNB': {}
}

transformers = {
    'CountVectorizer': CountVectorizer(),
    'TfidfTransformer': TfidfTransformer()
}

# will use default parameters for TfidfTransformer
transformer_params = {
    'CountVectorizer': {
        'min_df': [0.01, 0.05],
        'max_df': [0.4, 0.6, 0.8, 0.95],
        'max_features': [3000, 10000, 40000],
        'lowercase': [False],
        'ngram_range': [(1,3)],
    },
    'TfidfTransformer': {}
}

scoring = {
    'acc': 'accuracy'
}

docs = training_df['text']
nlp = spacy.load('en')

print('Tokenizing and lemmatizing...\n')
filtered_tokens = preprocess_tokens(docs, nlp)
print('Finished tokenizing and lemmatizing.\n')

X = [' '.join(tokens) for tokens in filtered_tokens]
y = training_df['target'].values

print('Performing grid search for', len(models.keys()), 'models with', len(transformers.keys()), 'transformers each.\n')
estimators = MultiClassifierGridSearchCV(models, model_params, transformers, transformer_params)

estimators.fit(X, y, cv=6, scoring=scoring, n_jobs=7)
print('Finished grid search.\n')

score_summary = estimators.score_summary()

print('Saving score summary...\n')
score_summary.to_csv('./model/score_summary.csv')
print('Score summary saved.\n')