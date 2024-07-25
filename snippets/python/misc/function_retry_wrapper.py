"""
TAGS: decorator|function|retry|wrapper
DESCRIPTION: A retry policy for a function call - allows the function to be recalled on error according to a defined retry policy
"""

import random
import time
from typing import Any, Callable, Iterable, Optional


class MaxRetriesExceededError(Exception):
    """Error raised if have retried too many times"""

    def __init__(self, message):
        super().__init__(message)


def retry_function_call(  # pylint: disable=too-many-arguments
    func: Callable,
    retry_pattern_seconds: Iterable[int | float | tuple[int | float, int | float]],
    exceptions_to_handle: tuple[Exception, ...],
    func_args: Optional[tuple] = None,
    func_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> Any:
    """
    Retries function (if it fails) according to retry pattern

    Args:
        func (Callable): The function to run
        retry_pattern_seconds (Iterable): Number of seconds to wait between failed function calls.
                                          Each entry in this Iterable can be an integer, float or tuple.
                                          Integer or float results in a deterministic wait time
                                          If tuple, then a random wait time is drawn (uniformly) from the range defined in the tuple
                                          e.g. retry_pattern_seconds=(1.5, (2,10)) will wait first 1.5 seconds and then a random
                                            number of seconds between 2 and 10
                                          After this wait time list is exhausted, exceptions will no longer be suppressed
        exceptions_to_handle (tuple[Exception, ...]): Exceptions which trigger a retry (exceptions not in this list will simply raise)
        func_args (tuple): unnamed arguments to pass to the function
        func_kwargs (dict): keyword (named) arguments to pass to the function
        verbose (bool): Print debugging information to standard out

    Returns:
        Any: If function executes without error, returns the function result

    Raises:
        MaxRetriesExceededError: if `retry_pattern_seconds` exhausted before successful function call

    Example:
        >>> import random
        >>> def random_failer(fail_prob: float) -> str:
        ...     '''Function which fails randomly with probability `fail_prob`'''
        ...     if random.uniform(0, 1) < fail_prob:
        ...         raise random.choice([ValueError, MemoryError])
        ...     return "function ran successfully"
        >>> func_output = retry_function_call(
        ...        random_failer,
        ...        func_args=(0.8,),
        ...        func_kwargs={},
        ...        retry_pattern_seconds=(0.1, (1,3), 2.5, (5,7.5),
        ...        exceptions_to_handle=(ValueError,),
        ...        verbose=True
        ...    )
        >>> print("Function output:", func_output)
    """
    if func_args is None:
        func_args = tuple()
    if func_kwargs is None:
        func_kwargs = {}

    for wait_n_seconds in retry_pattern_seconds:
        try:
            return func(*func_args, **func_kwargs)
        except Exception as err:  # pylint: disable=broad-exception-caught
            if verbose:
                print(f"received error {type(err)}")
            if any((isinstance(err, eh) for eh in exceptions_to_handle)):
                if isinstance(wait_n_seconds, tuple):
                    sleep_n_seconds = random.uniform(*wait_n_seconds)
                else:
                    sleep_n_seconds = wait_n_seconds
                if verbose:
                    print(f"waiting {sleep_n_seconds:,} seconds then retrying")
                time.sleep(sleep_n_seconds)
            else:
                raise err

    raise MaxRetriesExceededError("Exhausted retry_pattern_seconds")
