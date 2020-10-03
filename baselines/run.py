"""
    main script to run algorithms
"""

### imports: python
import sys
import os
import numpy as np


from common.cmd_util import common_arg_parser
import logger
import utils

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    print("can not import MPI")


def train(args, extra_args):
    """
    Main entrance to training session
    - group arguments for environment, agent and training
    - build environment
    - build model
    - train model with agent's learn() function

    Args:
        args: (argparse.ArgumentParser) parsed command line arguments
        extra_args: (dict) dictionary containing extra environment settings for agent
    """
    # environment type
    env_type, env_id = utils.get_env_type(args)
    print('env_type: {}'.format(env_type))
    # total timesteps
    total_timesteps = int(args.num_timesteps)
    # seed
    seed = args.seed
    # agent's learn function
    learn = utils.get_learn_function(args.alg)
    # default environment settings
    alg_kwargs = utils.get_learn_function_defaults(args.alg, env_type)
    # add (if any) extra settings from command line
    alg_kwargs.update(extra_args)

    # build environment
    env = utils.build_env(args)

    # save vectorized video recorder if specified (TBD)

    # add network type if specified
    if args.network:
        alg_kwargs['network'] = args.network
    # use default network type if not specified
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = utils.get_default_network(env_type)

    # summarize training setup
    print("Training {} on {}:{} with arguments \n{}".format(args.alg, env_type, env_id, alg_kwargs))

    # train model with the agent's learn() function
    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env



def main(args):
    """
    Main script
    - parse commandline arguments
    - set up logger
    - (optional) set up distributed training
    - enter training


    Known args:
        env: (str) environment id, e.g., 'Reacher-v2'
        env_type: (str) type of the environment, e.g., 'atari'
        seed: (int) random seed
        alg: (str) RL algorithm, e.g., 'ppo'
        num_timesteps: (float) number of total timesteps
        network: (str) network type for approximators, e.g., mlp, cnn
        gamestate: (??) the game state to load (used only for retro games)
        num_env: (int) number of environment copies to be run in parallel (if not specified, set to number of CPUs)
        reward_scale: (float) scale factor for reward (i.e., the discount factor?)
        save_path: (str) path to save trained model
        save_video_interval: (int) save video every x timesteps (0 means disable saving)
        save_video_length: (int) length in timesteps of recorded video
        log_path: (str) path to save learning curve data
        play: (bool) ??

    """

    # get commandline arguments
    arg_parser = common_arg_parser()
    args, _ = arg_parser.parse_known_args(args)
    extra_args = []

    # MPI for parallel computation (tricky to make it work on Win10)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        utils.configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        utils.configure_logger(args.log_path, format_strs=[])

    # train agent
    model, env = train(args, extra_args)

    # save trained agent
    if args.save_path is not None and rank == 0:
        save_path = os.path.expanduser(args.save_path)
        model.save(save_path)

    # test the trained agent by playing environment
    if args.play:
        logger.log("Running Trained Model")
        # reset environment
        obs = env.reset()
        # obtain initial state s0 if available
        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        # initialize episodic reward (or the empirical return G)
        # case for vectorized environments TBD
        episode_rew = np.zeros(1)

        # Q: should terminal condition for this loop really be indefinitely True?
        while True:
            # 1) make one policy step ??
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)
            # 2) make one environment step by taking action a
            # -- returns (observation, reward, terminal_state_flag, info_dict)
            obs, rew, done, _ = env.step(actions)
            # 3) accumulate episodic returns
            episode_rew += rew
            # 4) renders environment for displaying
            env.render()

            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print("episode_rew={}".format(episode_rew[i]))
                    episode_rew[i] = 0

    env.close()

    return model





if __name__ == '__main__':
    main(sys.argv)
