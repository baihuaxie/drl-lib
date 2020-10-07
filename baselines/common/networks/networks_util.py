"""
Utilities to register, build networks
"""

mapping = {}

# decorator
def register(name):
    """
    Decorator: register a network builder function in a dictionary 'mapping' by name: func

    Args:
        name: (str) network name; used as arguments passed to the decorator


    To register a new network builder:
    in another_network.py file, do:

    from networks_util import register
    @register(name='new_network_builder')
    def new_network_builder(**network_kwargs):
        ...
        return network_function_or_class_obj

    Note:
    - using a register decorator is a more Pythonic way to maintain & call dynamic functions
      compared to maintain a list of names
    """
    def _decorator(func):
        """
        inner decorator function
        - no need to modify behaviors so no wrappers
        """
        mapping[name] = func
        return func
    return _decorator



def get_network_builder(name):
    """
    Returns the network builer function by name

    Args:
        name: (str) name of network builder

    Returns:
        (network builder function) a network builder function that takes addition keyword arguments
                                     to instantiate a network class object

    Note:
    - call this function by: MyNetwork = get_network_builder(name)(**kwargs)
    """

    if callable(name):
        # if name is a callable function, return directly
        return name
    elif name in mapping.keys():
        # if name is str & registered as a network builder
        return mapping[name]
    else:
        # raise an error otherwise
        raise ValueError('Unknown network type: {}'.format(name))
