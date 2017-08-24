import random
import string


__author__     = ['Tom Van Mele <vanmelet@ethz.ch>', ]
__copyright__  = 'Copyright 2014, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'vanmelet@ethz.ch'


__all__ = [
    'random_name',
]


def random_name(n=17):
    return ''.join(random.choice(string.lowercase) for _ in range(n))


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":
    pass
