import pandas as pd
import numpy as np

from utils import get_tuple_list, get_pipeline_params
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

"""
Original code by David S. Batista (http://www.davidsbatista.net/blog/2018/02/23/model_optimization/)

I extended the code by adding the functionality to use multiple transformer steps and scoring functions during the grid
search. Furthermore, I added all the comments.
"""


class MultiClassifierGridSearchCV:
    """
    A class to do grid search using a pipeline that allows multiple transformer steps, classifiers, and scoring
    functions.

    ...

    Attributes
    ----------
    models: dict
        A dictionary of models where the key is the model name and the value is the model object.
    model_params: dict
        A dictionary of model parameters.
    model_keys: dict_keys
        The keys of the models dictionary.
    transformers: dict
        A dictionary of transformers where the key is the transformer name and the value is the transformer object.
    transformer_params: dict
        A dictionary of transformer parameters.
    transformer_keys: dict_keys
        The keys of the transformers dictionary.
    transformer_tuple_list: list of tuple
        A list containing the tuples (transformer_key, transformers[transformer_key]).
    grid_searches: dict
        A dictionary where the keys are the model keys and the values are the sklearn.model_selection.GridSearchCV
        objects that correspond to performing a grid search using the model given by the key.

    Methods
    -------
    fit(X, y, cv=5, n_jobs=-1, verbose=1, scoring=None, refit=False, return_train_score=False):
        Performs grid search on a given model.
    score_summary():
        Returns a pandas.DataFrame with the summary of the grid search of each model.
    """

    def __init__(self, models, model_params, transformers, transformer_params):
        """
        Constructs all the necessary attributes for the multi-classifier grid search object.
        :param models: dict
            Dictionary of model objects.
        :param model_params:
            Dictionary of parameter values for each model.
        :param transformers:
            Dictionary of transformer objects.
        :param transformer_params:
            Dictionary of parameter values for each transformer.
        """

        if not set(models.keys()).issubset(set(model_params.keys())):
            missing_params = list(set(models.keys()) - set(model_params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)

        if not set(transformers.keys()).issubset(set(transformer_params.keys())):
            missing_params = list(set(transformers.keys()) - set(transformer_params.keys()))
            raise ValueError("Some transformers are missing parameters: %s" % missing_params)

        if not all(isinstance(x, dict) for x in model_params.values()):
            wrong_model_params = [k for k, v in model_params.items() if not isinstance(v, dict)]
            raise TypeError("Values in the dictionary must be of type 'dict'. "
                            "These models have more than one parameter set: %s. "
                            "Please separate them into different models, e.g. 'model_param_set_1', 'model_param_set_2',"
                            " etc." % wrong_model_params)

        self.models = models
        self.model_params = model_params
        self.model_keys = models.keys()
        self.transformers = transformers
        self.transformer_params = transformer_params
        self.transformer_keys = transformers.keys()
        self.transformer_tuple_list = get_tuple_list(transformers)
        self.grid_searches = {}

    def fit(self, X, y, cv=5, n_jobs=-1, verbose=1, scoring=None, refit=False, return_train_score=False):
        """
        Fits the training data to a grid search object for every model. Runs the grid search for each model.

        :param X: numpy.ndarray
            Training data matrix.
        :param y: numpy.ndarray
            Labels for the training data.
        :param cv: int, cross-validation generator, or an interable (default = 5)
            Determines the cross-validation strategy.
        :param n_jobs: int (default = -1)
            Number of jobs to run in parallel. None means 1 and -1 means use all processors.
        :param verbose: int (default = 1)
            Controls the verbosity: the higher, the more messages.
        :param scoring: str, callable, list/tuple, or dict (default = None)
            A single str or a callable for single metric evaluation. For multiple metric evaluation, either give a list
            of (unique) strings or a dict with names as keys and callables as values.
        :param refit: bool (default = False)
            Refit an estimator using the best found parameters on the whole dataset.
        :param return_train_score: bool (default = False)
            If False, the cv_results_ attribute will not include training scores.
        :return None
        """

        for model_key in self.model_keys:
            print("Running GridSearchCV for %s." % model_key)
            model = self.models[model_key]

            model_pipe_params = get_pipeline_params({model_key: self.model_params[model_key]})
            pipe_params = get_pipeline_params(self.transformer_params)
            pipe_params.update(model_pipe_params)

            pipe_list = self.transformer_tuple_list + [(model_key, model)]
            pipeline = Pipeline(pipe_list)
            gs = GridSearchCV(pipeline, pipe_params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=return_train_score)
            gs.fit(X, y)
            self.grid_searches[model_key] = gs

    def score_summary(self):
        """
        Creates a pandas.DataFrame with the test score summary for each parameter combination and scoring function.

        :return df: pandas.DataFrame
            Contains the score summary for the run of the grid search.
        """

        def row(model_key, scorers, gs_params):
            """
            Defines a pandas.Series with the min, max, mean, and std score of the scorer function.

            :param model_key: str
                The name of the estimator (model).
            :param scorers: dict
                Dictionary of scorers containing their min, max, mean, and std scores.
            :param gs_params: dict
                Dictionary of transformer and model parameters.
            :return row_series: pandas.Series
                A pandas.Series that contains the transformer and model parameters as well as the scores of each scorer
                function.
            """
            gs_params.update({'estimator': model_key})
            row_series = pd.Series({**gs_params})

            # append the scores of each scorer to row_series
            for sc in scorers:
                min_score_key = sc + '_min_score'
                max_score_key = sc + '_max_score'
                mean_score_key = sc + '_mean_score'
                std_score_key = sc + '_std_score'
                d = {
                    min_score_key: scorers[sc]['min_score'],
                    max_score_key: scorers[sc]['max_score'],
                    mean_score_key: scorers[sc]['mean_score'],
                    std_score_key: scorers[sc]['std_score'],
                }
                row_series = pd.concat([row_series, pd.Series({**d})])
            return row_series

        rows = []
        for k in self.grid_searches:
            params = self.grid_searches[k].cv_results_['params']
            scores = {}
            for scorer in self.grid_searches[k].scoring:
                cv_results = self.grid_searches[k].cv_results_
                mean_score = cv_results['mean_test_' + scorer]
                std_score = cv_results['std_test_' + scorer]
                min_score = np.ones(len(params))
                max_score = np.zeros(len(params))

                # get the minimum and maximum scores across all splits
                for i in range(self.grid_searches[k].cv):
                    key = "split{}_test_{}".format(i, scorer)
                    r = cv_results[key]
                    min_score = np.minimum(min_score, r)
                    max_score = np.maximum(max_score, r)

                scorer_dict = {
                    scorer: {
                        'min_score': min_score,
                        'max_score': max_score,
                        'mean_score': mean_score,
                        'std_score': std_score
                    }
                }
                scores.update(scorer_dict)

            for i, p in enumerate(params):
                # need to get the scores that correspond to the correct parameters, hence the sub_v[i]
                sc_dict = {k: {sub_k: sub_v[i] for sub_k, sub_v in v.items()} for k, v in scores.items()}
                rows.append(row(k, sc_dict, p))

        df = pd.concat(rows, axis=1).T

        # only the columns for the estimator, parameters (__), and scores (_score) need to be kept
        columns = ['estimator'] + [s for s in df.columns if '__' in s] + [s for s in df.columns if '_score' in s]
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
