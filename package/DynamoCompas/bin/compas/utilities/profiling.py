from __future__ import print_function

import cStringIO
import cProfile
import pstats

from functools import wraps


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = [
    'print_profile',
]


def print_profile(func):
    """Decorate a function with automatic profile printing.

    Parameters:
        func (function) : The function to decorate.

    Returns:
        function : The decorated function.

    Examples:

        .. code-block:: python

            @print_profile
            def f(n):
                \"\"\"Sum up all integers below n.\"\"\"
                return sum(for i in range(n))

            print(f(100))
            print(f.__doc__)
            print(f.__name__)

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profile = cProfile.Profile()
        profile.enable()
        #
        res = func(*args, **kwargs)
        #
        profile.disable()
        stream = cStringIO.StringIO()
        stats  = pstats.Stats(profile, stream=stream)
        stats.strip_dirs()
        stats.sort_stats(1)
        stats.print_stats(20)
        print(stream.getvalue())
        #
        return res
    return wrapper


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":

    @print_profile
    def f(n):
        """sum all integers below n"""
        s = 0
        for i in range(n):
            s += i
        return s

    print(f(100))

    print(f.__doc__)
    print(f.__name__)
