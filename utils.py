import pandas as pd
import numpy as np
import functools
import gc
import itertools
import sys
from timeit import default_timer as _timer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def get_pipeline_params(param_dict):
    new_params = {}
    for key in param_dict.keys():
        for param_key in param_dict[key].keys():
            new_key = key + '__' + param_key
            new_value = tuple(param_dict[key][param_key])
            new_params.update({new_key: new_value})

    return new_params


def get_tuple_list(d):
    tuple_list = []
    for key in d.keys():
        tuple_list.append((key, d[key]))

    return tuple_list


def get_tuple_dict(d):
    tuple_dict = {}
    for key in d.keys():
        tuple_dict.update({key: d[key]})

    return tuple_dict


class EstimatorSelectionHelper:

    def __init__(self, models, model_params, transformers, transformer_params):
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
        self.model_tuple_dict = get_tuple_dict(models)
        self.transformers = transformers
        self.transformer_params = transformer_params
        self.transformer_keys = transformers.keys()
        self.transformer_tuple_list = get_tuple_list(transformers)
        self.grid_searches = {}

    def fit(self, X, y, cv=5, n_jobs=-1, verbose=1, scoring=None, refit=False, return_train_score=True):
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
            gs.fit(X,y)
            self.grid_searches[model_key] = gs

    def score_summary(self):
        def row(key, scores, params):
            params.update({'estimator': key})
            row_series = pd.Series({**params})
            for scorer in scores:
                min_score = scorer + '_min_score'
                max_score = scorer + '_max_score'
                mean_score = scorer + '_mean_score'
                std_score = scorer + '_std_score'
                d = {
                     min_score: scores[scorer]['min_score'],
                     max_score: scores[scorer]['max_score'],
                     mean_score: scores[scorer]['mean_score'],
                     std_score: scores[scorer]['std_score'],
                }
                row_series = pd.concat([row_series, pd.Series({**d})])
            return row_series

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = {}
            for scorer in self.grid_searches[k].scoring:
                cv_results = self.grid_searches[k].cv_results_
                mean_score = cv_results['mean_test_' + scorer]
                std_score = cv_results['std_test_' + scorer]
                min_score = np.ones(len(params))
                max_score = np.zeros(len(params))

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
                sc_dict = {k: {sub_k: sub_v[i] for sub_k, sub_v in v.items()} for k, v in scores.items()}
                rows.append(row(k, sc_dict, p))

        df = pd.concat(rows, axis=1).T

        columns = ['estimator'] + [s for s in df.columns if '__' in s] + [s for s in df.columns if '_score' in s]
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


def timeit(_func=None, *, repeat=3, number=1000, file=sys.stdout):
    """Decorator: prints time from best of `repeat` trials.
    Mimics `timeit.repeat()`, but avg. time is printed.
    Returns function result and prints time.
    You can decorate with or without parentheses, as in
    Python's @dataclass class decorator.
    kwargs are passed to `print()`.
    >>> @timeit
    ... def f():
    ...     return "-".join(str(n) for n in range(100))
    ...
    >>> @timeit(number=100000)
    ... def g():
    ...     return "-".join(str(n) for n in range(10))
    ...
    """

    _repeat = functools.partial(itertools.repeat, None)

    def wrap(func):
        @functools.wraps(func)
        def _timeit(*args, **kwargs):
            # Temporarily turn off garbage collection during the timing.
            # Makes independent timings more comparable.
            # If it was originally enabled, switch it back on afterwards.
            gcold = gc.isenabled()
            gc.disable()

            try:
                # Outer loop - the number of repeats.
                trials = []
                for _ in _repeat(repeat):
                    # Inner loop - the number of calls within each repeat.
                    total = 0
                    for _ in _repeat(number):
                        start = _timer()
                        result = func(*args, **kwargs)
                        end = _timer()
                        total += end - start
                    trials.append(total)

                # We want the *average time* from the *best* trial.
                # For more on this methodology, see the docs for
                # Python's `timeit` module.
                #
                # "In a typical case, the lowest value gives a lower bound
                # for how fast your machine can run the given code snippet;
                # higher values in the result vector are typically not
                # caused by variability in Python’s speed, but by other
                # processes interfering with your timing accuracy."
                best = min(trials) / number
                print(
                    "Best of {} trials with {} function"
                    " calls per trial:".format(repeat, number)
                )
                print(
                    "Function `{}` ran in average"
                    " of {:0.3f} seconds.".format(func.__name__, best),
                    end="\n\n",
                    file=file,
                )
            finally:
                if gcold:
                    gc.enable()
            # Result is returned *only once*
            return result

        return _timeit

    # Syntax trick from Python @dataclass
    if _func is None:
        return wrap
    else:
        return wrap(_func)