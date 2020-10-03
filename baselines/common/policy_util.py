"""
Encapsulation of policy and value function estimations with shared parameters
"""


class PolicyWithValue(object):
    """
    Common interface for policy and value functions
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None)