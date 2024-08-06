"""
TAGS: decorate|decorating|decorator|log|logger|logging|wrap|wrapper
DESCRIPTION: Decorator used to log function calling (in order to declutter the calling code)
USAGE: See end of this script (if __name__=="__main__") for example usage
"""

import logging
from functools import wraps
from typing import Callable


def log_function_call(logging_func: Callable, log_args: bool):
    """Decorator logging the function/method call (possibly including arguments) using the provided `logging_func`

    Args:
        logging_func (Callable): A logging function (or method) that can be called with non-named arguments (e.g. logging.info)
        log_args (bool): If true, also logs args and kwargs in called function
    """

    def wrapper_func(wrapped_func: Callable) -> Callable:
        @wraps(wrapped_func)
        def inner_func(*args, **kwargs):
            if log_args:
                logging_func(
                    "Called %s(%s)",
                    wrapped_func.__name__,
                    ", ".join(
                        [
                            (
                                f"'{str(arg)[:100]}...'"
                                if len(str(arg)) > 100
                                else f"'{arg}'" if isinstance(arg, str) else str(arg)
                            )
                            for arg in args
                        ]
                        + [
                            (
                                f"{k}='{str(v)[:100]}...'"
                                if len(str(v)) > 100
                                else f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
                            )
                            for k, v in kwargs.items()
                        ]
                    ),
                )
            else:
                logging_func("Called %s()", wrapped_func.__name__)
            return wrapped_func(*args, **kwargs)

        return inner_func

    return wrapper_func


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    @log_function_call(logger.info, log_args=True)
    def get_data(data_type: str, ignore_up_to_n_errors: int) -> str:
        return f"some {data_type} data"

    @log_function_call(logger.info, log_args=True)
    def process_data(data: str, operation: str) -> str:
        return f"{operation}ed {data}"

    @log_function_call(logger.info, log_args=True)
    def export_data(output_path: str, suppress_errors: bool) -> int:
        return 200

    raw_data = get_data("parquet", 5)
    processed_data = process_data(raw_data, operation="depersonalise")
    status_code = export_data(
        "gs://depersonalised_data/output.parquet", suppress_errors=True
    )
