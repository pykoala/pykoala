import warnings
from functools import wraps
from typing import Iterable, Optional, Type, Union

WarningType = Type[Warning]

def suppress_warnings(
    _func=None,
    *,
    categories: Optional[Union[WarningType, Iterable[WarningType]]] = None,
    message: str = "",
    module: str = "",
    lineno: int = 0,
):
    """
    Decorator to suppress warnings emitted inside a function.

    Usage
    -----
    @suppress_warnings
    def f(...): ...

    @suppress_warnings(categories=UserWarning)
    def g(...): ...

    @suppress_warnings(categories=(UserWarning, RuntimeWarning), message=".*deprecated.*")
    def h(...): ...

    Parameters
    ----------
    categories
        A warning class or iterable of classes to suppress. If None, all warnings are ignored.
    message
        Optional regex to match the warning message (same semantics as warnings.filterwarnings).
    module
        Optional regex to match the module name that issued the warning.
    lineno
        Optional line number to match.

    Notes
    -----
    - Suppression is confined to the wrapped call via warnings.catch_warnings.
    - Does not affect warnings outside the function body.
    """
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                if categories is None:
                    warnings.simplefilter("ignore")
                else:
                    cats = categories
                    if not isinstance(cats, (tuple, list)):
                        cats = (cats,)
                    for cat in cats:
                        warnings.filterwarnings(
                            "ignore",
                            message=message,
                            category=cat,
                            module=module,
                            lineno=lineno,
                        )
                return func(*args, **kwargs)
        return _wrapper

    return _decorator if _func is None else _decorator(_func)