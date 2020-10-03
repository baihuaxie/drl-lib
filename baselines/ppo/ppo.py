"""
Proximal Policy Gradient algorithm (https://arxiv.org/abs/1707.06347)
"""

import os

from baselines.common.misc_util import set_global_seed




def learn(*, network, evn, total_timesteps, eval_env=None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95, log_interval=10, minibatch=4, noptepochs=4,
          cliprange=0.2, save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None,
          mpi_rank_weight=1):
    """
    Learn policy by the PPO algorithm

    Args:

        seed: (float) random seed for reproducibility

    """
    # set global random seed
    set_global_seed(seed)

    total_timesteps = int(total_timesteps)

    