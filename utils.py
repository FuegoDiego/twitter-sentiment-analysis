import functools
import gc
import itertools
import sys

from timeit import default_timer as _timer


def get_pipeline_params(param_dict):
    """
    Flattens a nested dictionary. The new keys are a concatenation of the original parent keys and their corresponding
    child keys.
    :param
        param_dict (dict): a nested dictionary
    :return
        new_params (dict): the result of flattening param_dict
    """
    new_params = {}
    for key in param_dict.keys():
        for param_key in param_dict[key].keys():
            new_key = key + '__' + param_key
            new_value = tuple(param_dict[key][param_key])
            new_params.update({new_key: new_value})

    return new_params


def get_tuple_list(d):
    """
    Creates a list of tuples where each tuple is of the form (key, value) where key is a key in the dictionary d and
    value is the corresponding value of the key in d.
    :param d: dict
        A dictionary
    :return tuple_list: list
        A list of tuples of the form (key, d[key]) where key is a key in d
    """
    tuple_list = []
    for key in d.keys():
        tuple_list.append((key, d[key]))

    return tuple_list


def print_progressbar(i, n):
    """Prints a progressbar for the ith step in a series of n steps."""
    sys.stdout.write('\r')
    sys.stdout.write("[{:{}}] {:.1f}%".format("=" * i, n - 1, (100 / (n - 1) * i)))
    sys.stdout.flush()


def timeit(_func=None, *, repeat=3, number=1000, file=sys.stdout):
    """Decorator: prints time from best of `repeat` trials.
    Mimics `timeit.repeat()`, but avg. time is printed.
    Returns function result and prints time.
    You can decorate with or without parentheses, as in
    Python's @dataclass class decorator.
    kwargs are passed to `print()`.

    Examples
    --------

    @timeit
    def f():
         return "-".join(str(n) for n in range(100))

    @timeit(number=100000)
     def g():
         return "-".join(str(n) for n in range(10))
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
