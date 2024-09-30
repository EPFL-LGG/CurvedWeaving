from _benchmark import *
import functools
import time

def benchmarkit(fn):
    # Ensure that the name and docstring of 'fn' is preserved in 'wrapper'
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      # the wrapper passes all parameters to the function being decorated
      start_timer_section(fn.__name__)
      res = fn(*args, **kwargs)
      stop_timer_section(fn.__name__)
      return res
    return wrapper
