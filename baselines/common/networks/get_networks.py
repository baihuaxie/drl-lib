"""
Utilities to register, build networks
"""

from baselines.common.networks.cnn import _convnet, BasicBlock
from baselines.common.networks.mlp import MLP

mapping = {}

# decorator
def register(name):
    """
    Decorator: register a network builder function in a dictionary 'mapping' by name: func

    Args:
        name: (str) network name; used as arguments passed to the decorator


    Note:
    - a network builder is a function that takes **kwargs arguments and returns a customized network object
        - i.e., call this function by MyNetwork = network_func(**kwargs)
    - using a register decorator is a more Pythonic way to maintain & call dynamic functions compared to
      maintain a list of names

    To register a new network builder:
    in another_network.py file, do:

    from networks_util import register
    @register(name='new_network_builder')
    def new_network_builder(**network_kwargs):
        ...
        return network_function_or_class_obj

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
        (a customized network object) a network object that is customized by **kwarg arguments

    Note:
    - call this function by: MyNetwork = get_network_builder(name)(**kwargs)
    - use the return of this function by: network_output = MyNetwork(network_input)
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


@register(name='simplecnn')
def convnet_simplecnn_k3s4(pretrained=False, progress=False, **kwargs):
    """
    Simple CNN network
    - kernel_size = 3 (default)
    - stacks = 4
        each stack contains 1 block with stride=2
    - latent_dim = 512
    """
    return _convnet('simplecnn', block=BasicBlock, layers=[1, 1, 1, 1], latent_dim=512,
                    pretrained=pretrained, progress=progress, **kwargs)


@register(name='mlp')
def mlp(obs_dim, act_dim):
    """
    2-layer MLP
    """
    return MLP(obs_dim, act_dim)