"""
Utilities for command line
"""

import argparse


def arg_parser():
    """
    Returns an empty argparse.ArgumentParser object
    """
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """
    Returns an argparse.ArgumentParser object with common arguments for atari and mujoco environments
    """

    parser = arg_parser()
    parser.add_argument("--env", help="environment ID", type=str, default="Reacher-v2")
    parser.add_argument("--env_type", help="type of environment", type=str, default=None)
    parser.add_argument("--seed", help="random number generator seed", type=int, default=None)
    parser.add_argument("--alg", help="algorithm", type=str, default="ppo")
    parser.add_argument("--num_timesteps", help="maximum number of time steps per iteration", type=float, default=1e6)
    parser.add_argument("--network", help="network type for function approximation, (mlp, cnn, lstm, cnn_lstm, conv_only)", type=str, default=None)
    parser.add_argument("--gamestate", help="game state to load", type=str, default=None)
    parser.add_argument("--num_env", help="number of environment copies run in parallel", type=int, default=None)
    parser.add_argument("--reward_scale", help="reward scale factor", type=float, default=1.0)
    parser.add_argument("--save_path", help="path to save trained model", type=str, default=None)
    parser.add_argument("--save_video_interval", help="save video every x steps", type=int, default=0)
    parser.add_argument("--save_video_length", help="length of recorded video in seconds", type=int, default=200)
    parser.add_argument("--log_path", help="directory to save learning curve data", type=str, default=None)
    parser.add_argument("--play", help="", default=False, action="store_true")
    
    return parser

