"""
    main script to run algorithms
"""

### imports: python
import sys

import gym

from common.cmd_util import common_arg_parser
import utils

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    print("can not import MPI")


def train(args, extra_args):
    """
    Main training loop

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
    alg_kwargs.update(extra_args)

    env = utils.build_env(args)



def main(args):
    """
    Main function

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

    model, env = train(args, extra_args)







if __name__ == '__main__':
    main(sys.argv)