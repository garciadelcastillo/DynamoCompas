__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


def get_axes_dimension(axes):
    if hasattr(axes, 'get_zlim'):
        return 3
    else:
        return 2


def assert_axes_dimension(axes, dim):
    assert get_axes_dimension(axes) == dim, 'The provided axes are not {0}D.'.format(dim)


def width_to_dict(width, dictkeys, defval=None):
    width = width or defval
    if isinstance(width, (int, float)):
        return dict((key, width) for key in dictkeys)
    if isinstance(width, dict):
        for k, w in width.items():
            if isinstance(w, (int, float)):
                width[k] = w
        return dict((key, width.get(key, defval)) for key in dictkeys)
    raise Exception('This is not a valid width format: {0}'.format(type(width)))


def size_to_sizedict(size, dictkeys, defval=None):
    size = size or defval
    if isinstance(size, (int, float)):
        return dict((key, size) for key in dictkeys)
    if isinstance(size, dict):
        for k, s in size.items():
            if isinstance(s, (int, float)):
                size[k] = s
        return dict((key, size.get(key, defval)) for key in dictkeys)
    raise Exception('This is not a valid size format: {0}'.format(type(size)))


def synchronize_scale_axes(axes):
    pass


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":
    pass
