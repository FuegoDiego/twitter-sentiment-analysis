import pandas as pd
import spacy
from process_tokenize import preprocess_tokens
from utils import EstimatorSelectionHelper
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer

# load the data
training_df = pd.read_csv('./data/tweets_training_data.csv',
                          header=None,
                          names=['target', 'id', 'date', 'flag', 'user', 'text'],
                          usecols=['target', 'id', 'date', 'text'],
                          encoding='latin-1')

training_df = training_df.groupby('target').head(1000).reset_index(drop=True)
training_df = training_df.sample(frac=1).reset_index(drop=True)

models = {
    'SVC': SVC(),
    'SGDClassifier': SGDClassifier()
}

model_params = {
    'SVC': {'kernel': ['rbf'], 'C': [1], 'gamma': [0.001]},
    'SGDClassifier': {'max_iter': [20], 'alpha': [0.00001], 'penalty': ['elasticnet']}
}

transformers = {
    'CountVectorizer': CountVectorizer(),
    'TfidfTransformer': TfidfTransformer()
}

transformer_params = {
    'CountVectorizer': {'max_df': [0.5], 'max_features': [20]},
    'TfidfTransformer': {'use_idf': [True, False]}
}

docs = training_df['text']
nlp = spacy.load('en')

filtered_tokens = preprocess_tokens(docs, nlp)
X = [' '.join(tokens) for tokens in filtered_tokens]
y = training_df['target'].values
scoring = {
    'f1_scorer': make_scorer(f1_score, pos_label=4),
    'recall_scorer': make_scorer(recall_score, pos_label=4),
    'precision_scorer': make_scorer(precision_score, pos_label=4),
    'acc': 'accuracy'
}

helper = EstimatorSelectionHelper(models, model_params, transformers, transformer_params)

helper.fit(X, y, cv=3, scoring=scoring)

helper.score_summary()

print()