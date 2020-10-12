"""
Main loop for Proximal Policy Gradient (PPO) algorithm (https://arxiv.org/abs/1707.06347)
"""

import os
from time import perf_counter
import numpy as np

from baselines.common import logger
from baselines.common.misc_util import set_global_seed
from baselines.common.math_util import explained_variance
from baselines.common.policy_util import build_policy
from baselines.ppo.runner import Runner
from baselines.ppo.model import Model

try:
    import MPI
except ImportError:
    MPI = None




def learn(*, network, env, total_timesteps, eval_env=None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95, log_interval=10, minibatches=4, noptepochs=4,
          cliprange=0.2, save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None,
          mpi_rank_weight=1, **network_kwargs):
    """
    Main interface function to learn policy by the PPO algorithm

    Args:
        network:            (str) policy network architecture; e.g., mlp, lstm, cnn, cnn_lstm, etc.
        env:                (Env class object) environment
        seed:               (float) random seed for reproducibility
        nsteps:             (int) number of timesteps of the vectorized environments per training iteration
                            i.e., batch_size (in # of steps) = nstep * num_envs
        minibatches:        (int) number of mini-batches per training iteration
        total_timesteps:    (int) total number of timesteps (i.e., actions) taken in the environment for the training session
        noptepochs:         (int) number of epochs (one update per epoch) per training iterations
        log_interval:       (int) number of timesteps between logging events
        **network_kwargs:   pointer to arguments for policy / value network builder

    Notes:
    1) batch_size, epochs, iterations:
    -- each new iteration would use a new batch_size of training data gathered by running the environment & collecting statistics
        - each batch_size of training data is used to do multiple updates per iteration
    -- example: nsteps = 2048, num_envs = 1 (no parallel copies)
    -- batch_size = 2048; means 2048 environment steps for each training iteration
    -- let minibatches = 4; means batch_size_train = 2048 / 4 = 512 timesteps for each training update
    -- let noptepochs = 4
    -- hence for one training iteration:
        - has 4 epochs
        - each epoch has 4 minibatches
        - each minibatch has 512 timesteps
        - make one update per minibatch
        - total updates per iteration is thus 4*4 = 16 with total of 2048 timesteps as trainng data

    """
    ### ----------------------------------------------------------------------------------------------------- ###
    ### step 1: initialize training settings
    ### ----------------------------------------------------------------------------------------------------- ###

    # set global random seed (for distributed setting)
    set_global_seed(seed)
    # check if MPI is active
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # get total number of timesteps in a training session (hyperparameter)
    total_timesteps = int(total_timesteps)
    # get number of environment copies simulated in parallel
    num_envs = env.num_envs

    # calculate batch sizes
    # a) batch_size
    # -- measured in number of timesteps
    # -- is the number of timesteps for one training iteration (or timesteps per iteration in some papers)
    # -- = number of parallel environment copies x number of steps per environment copy per update (hyper-parameter)
    batch_size = num_envs * nsteps
    # b) batch_size_train
    # -- measured in number of timesteps
    # -- is the number of timesteps for one training update (multiple updates per iteration, specified by noptepochs)
    # -- = batch_size // minibatches (hyperparameter; is the number of minibatches needed per update)
    batch_size_train = batch_size // minibatches
    # make sure batch_size is divisible by minibatches
    assert batch_size % minibatches == 0, "batch_size is not divisible by minibatches"
    # c) num_iterations
    # -- total number of iterations specified for a training session
    # -- = total_timesteps // batch_size
    num_iterations = total_timesteps // batch_size

    # get state and action spaces
    ob_space = env.observation_space
    ac_space = env.action_space

    ### ----------------------------------------------------------------------------------------------------- ###
    ### step 2: instantiate PPO objects
    ### ----------------------------------------------------------------------------------------------------- ###

    # 1) build the policy + value networks by instantiating the policy object
    # -- policy class is in common & shared by different agents, with agent-specific customizations
    policy = build_policy(env, network, **network_kwargs)

    # 2) build the agent algorithm by instantiating the model object
    if model_fn is None:
        model_fn = Model

    model = model_fn()

    # load pretrained model if specified
    if load_path is not None:
        model.load(load_path)

    # 3) build the episode running framework by instantiating the runner object
    runner = Runner()
    if eval_env is not None:
        eval_runner = Runner()

    ### ----------------------------------------------------------------------------------------------------- ###
    ### step 3: main training loop
    ### ----------------------------------------------------------------------------------------------------- ###

    # start total timer
    t_first_start = perf_counter()

    # loop over training iterations
    for iteration in range(1, num_iterations+1):

        # start timer
        tstart = perf_counter()

        # schedule learning rate
        # schedule clip range

        # logging events
        if iteration % log_interval == 0 and is_mpi_root:
            logger.info("Stepping environment...")

        # 1. Get a minibatch of statistics
        # -- by runner.run()
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        # -- evaluation mode
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run()

        # logging events
        if iteration % log_interval == 0 and is_mpi_root:
            logger.info("Done.")

        # accumulate episode info dict

        # 2. Calculate loss for each minibatch and accumulate
        mb_lossvals = []
        # A) non-recurrent environments (states == None means at terminal state?)
        if states is None:
            # create an index array for each timestep in batch_size
            indices = np.arange(batch_size)
            # loop over epochs
            for _ in range(noptepochs):
                # randomize the indices
                np.random.shuffle(indices)
                # loop over the training data from 0 to batch_size in steps of batch_size_train
                for start in range(0, batch_size, batch_size_train):
                    # calculate end index
                    end = start + batch_size_train
                    # make an index slice of training data for a minibatch in size batch_size_train
                    mb_indices = indices[start:end]
                    # get a minibatch slice of training data
                    slices = (arr[mb_indices] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    # accumulate loss
                    mb_lossvals.append(model.train(*slices))
        # B) recurrent environments (no terminal states?)
        else:
            # Q: why assert this? this won't hold if num_envs = 1
            assert num_envs % minibatches ==  0, "num_envs is not divisible by minibatches"
            envs_per_batch = num_envs // minibatches
            # create an index array for each parallel environment copy
            env_indices = np.arange(num_envs)
            # create a flattened index matrix for all timesteps in all environment copies
            flat_indices = np.arange(num_envs * nsteps).reshape(num_envs, nsteps)
            # loop over epochs
            for _ in range(noptepochs):
                # randomize environment indices
                np.random.shuffle(env_indices)
                # loop over the environments from 0 to num_envs in steps of envs_per_batch
                for start in range(0, num_envs, envs_per_batch):
                    # calculate end environment index
                    end = start + envs_per_batch
                    # make an index slice of environments in size envs_per_batch
                    mb_env_indices = env_indices[start:end]
                    # make an index slice of flattened timesteps
                    mb_flat_indices = flat_indices[mb_env_indices].ravel()
                    # slice training data
                    slices = (arr[mb_flat_indices] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    # slice states
                    mb_states = states[mb_env_indices]
                    # accumulate loss
                    mb_lossvals.append(model.train(*slices, mb_states))

        # 3. Update networks
        # average training losses over minibatches
        lossvals = np.mean(mb_lossvals, axis=0)
        # end timer
        tnow = perf_counter()
        # calculate frames per second (fps)
        # - treat each timestep as one frame
        fps = int(batch_size / (tnow - tstart))

        if update_fn is not None:
            update_fn(iteration)

        # 4. Logistics
        # -- log events
        if iteration % log_interval == 0 or iteration == 1:

            ev = explained_variance(values, returns)

            logger.logkv("misc/serial_timesteps", iteration * nsteps)
            logger.logkv("misc/num_iterations", iteration)
            logger.logkv("misc/total_timesteps", iteration * batch_size)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))

            logger.logkv("misc/time_elapsed", tnow - t_first_start)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv("loss/" + lossname, lossval)

            logger.dumpkvs()

        # -- save checkpoints
        if save_interval and (iteration % save_interval == 0 or iteration == 1) and logger.get_dir() and is_mpi_root:
            checkdir = os.path.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = os.path.join(checkdir, '%.5i'%iteration)
            print("Saving to {}".format(savepath))
            model.save(savepath)

    return model
