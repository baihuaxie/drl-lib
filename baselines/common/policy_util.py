"""
Encapsulation of policy and value function estimations with shared parameters
"""


class PolicyWithValue(object):
    """
    Common interface for policy and value networks
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None):
        """
        Constructor
        """


def build_policy(env, policy_network, value_network=None, **policy_kwargs):
    """
    Policy / Value network builder
    Returns a PolicyWithValue class object
    """