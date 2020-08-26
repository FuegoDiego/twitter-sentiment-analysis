# Twitter Sentiment Analysis
This project is part of an ongoing expansion of my Data Science skills as well as my portfolio. I really enjoyed learning about NLP and creating a solution that allows you to train multiple models with cross-validation using a scikit-learn pipeline.

## Project Intro/Objective
The purpose of this project is to improve my Data Science skills while I look for professional opportunities in this field. More specifically, it was meant to improve my understanding of binary classification, NLP, data preprocessing, cross-validation, model selection, and pipelines.

### Methods Used
* Natural Language Processing
* Machine Learning
* Binary Classification
* Cross-Validation
* Pipeline
* Tokenization
* Lemmatization
* Data Preprocessing

### Technologies
* Python
* Pandas, Numpy
* Spacy
* Scikit-Learn

## Project Description
The goal of the project is to train multiple models using using a pipeline to detect sentiment in tweets. The data used for this project is the [Sentiment140](http://help.sentiment140.com/for-students) dataset, but you can use any dataset of your choice. 

I used the __spacy__ library to tokenize and lemmatize the tweets in a subset of the dataset (50,000 tweets for each positive/negative sentiment). The pipeline class in the __scikit-learn__ library allows you to perform a grid-search using cross-validation for a single model. It also allows you to perform any number of transformation steps before training the model. Furthermore, it allows the use of multiple scoring methods to gauge model performance. For this project, I wanted to be able to perform any number of transformation steps followed by _multiple_ classifiers with multiple scoring methods. 

I found an extension to this pipeline by [David S. Batista](http://www.davidsbatista.net/blog/2018/02/23/model_optimization/) that allowed you to perform cross-validation on multiple models. However, his solution assumed the data was already transformed and only had the capability of using a single scoring method. In order to achieve my goal, I extended his code and now you can create a pipeline that accomodates multiple transformation steps, classifiers, and scoring methods.

After increasing the capabilities of the pipeline, I created a pipeline that included the following transformations, classifiers, and scoring functions using the __scikit-learn__ library.

_Transformations_
* CountVectorizer
* TfidfTransformer

_Classifiers_
* Multinomial Naive Bayes
* Linear SVM
* Logistic Regression

_Scoring Functions_
* Accuracy

The pipeline was run using 100,000 data points, and took over 6 hours to complete. I am currently facing a performance block given the capabilities of the machine I am running it on. I specify the tool versions and my machine's specs in a later section.

## Getting Started

1. Clone this repo.
2. Put your dataset in the `data` directory and rename it to __tweets_training_data.csv__. _NOTE_: make sure your data is in CSV format and that the labels are in a column called _target_. 
3. Modify the `main.py` script.
   - Modify the `models`, `model_params`, `transformers`, `transformer_params`, and `scoring` dictionaries to use     transformations, models, and scoring functions of your choice. _NOTE_: please don't change the name of any of the variables.
   - Comment out the line `training_df = training_df.groupby('target').head(50000).reset_index(drop=True)` unless you want to use 50,000 data points for each target variable.
   - Modify the line `estimators.fit(X, y, cv=7, scoring=scoring, n_jobs=7)` if you want to use a different value for the k-folds `cv` and number of concurrent workers `n_jobs`.
 4. Execute the `main.py` script.
 5. Check the `model` directory. There should be a CSV file `score_summary.csv` that contains a table with the results of the grid search that you can use for your analysis.

## Example
For this example, I ran the following configuration:
```python
models = {
    'MultinomialNB': MultinomialNB(),
    'SGDClassifier': SGDClassifier(),
    'LogisticRegression': LogisticRegression()
}

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

estimators.fit(X, y, cv=3, scoring=scoring, n_jobs=7)
```
I only used accuracy as a scoring function given that my data was balanced. For this run, only three folds were used for cross-validation. After execution, the models with the best accuracy were all permutations of Logistic Regression. The models with the highest mean accuracy accross all cross-validation splits were chosen as the best models. The following table shows the parameter values and scores for the best 5 models out of 4,824 along with the transformer parameter values.

Model | CountVectorizer__max_df | CountVectorizer__max_features | CountVectorizer__min_df | CountVectorizer__ngram_range | LogisticRegression__C | LogisticRegression__l1_ratio | LogisticRegression__max_iter | LogisticRegression__penalty | LogisticRegression__solver | acc_mean_score
----- | ----------------------- | ----------------------------- | ----------------------- | ---------------------------- | --------------------- | ---------------------------- | ---------------------------- | --------------------------- | -------------------------- | --------------
LogisticRegression | 0.4 | 10000 | 0.01 | (1, 3) | 0.5 | 0.5 | 100.0 | 7.0 | elasticnet | saga | 0.6517499897211945
LogisticRegression | 0.95 | 3000 | 0.01 | (1, 3) | 1.0 | 1.0 | 5000.0 | 7.0 | elasticnet | saga | 0.6517499885211704
LogisticRegression | 0.95 | 40000 | 0.01 | (1, 3) | 0.5 | 0.5 | 100.0 | 7.0 | elasticnet | saga | 0.6517399899211904
LogisticRegression | 0.8 | 10000 | 0.01 | (1, 3) | 0.5 | 0.5 | 100.0 | 7.0 | elasticnet | saga | 0.6517399899211904
LogisticRegression | 0.4 | 40000 | 0.01 | (1, 3) | 0.5 | 0.5 | 1000.0 | 7.0 | elasticnet | saga | 0.6517399899211904

The best model achieved a mean accuracy of 65.175%. There clearly is room for improvement such as doing a more extensive grid search, increasing the number of folds in the cross-validation, and using more training data. I am currently working on training a Logistic Regression model with more exhaustive parameter combinations and 7-fold cross-validation (7-fold sets aside about 15% of the data for testing). I expect the results to improve and I will update this section accordingly once that's done.


## Python and Library Versions
* Python `3.7.7`
* Pandas `1.0.5`
* Numpy `1.19.1`
* Spacy `2.3.2`
* sklearn `0.23.1`

## Machine Specifications
* OS: `Microsoft Windows 10 Home Version 1903`
* Processor: `Intel(R) Core(TM) i5-8250U` with `4` cores and `8` logical processors
 * RAM: `8 GB`
 * CUDA support: No

## Contact
Please feel free to contact me if you found this interesting and think I would be a good fit at your company :)
