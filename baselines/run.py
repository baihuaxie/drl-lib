"""
    main script to run algorithms
"""

### imports

import sys


from common.cmd_util import common_arg_parser

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    print("can not import MPI")

def main(args):
    """
        main function
    """
    
    # get commandline arguments
    arg_parser = common_arg_parser()
    args, _ = arg_parser.parse_known_args(args)

    # MPI for parallel computation (tricky to make it work on Win10)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])







if __name__ == '__main__':
    main(sys.argv)