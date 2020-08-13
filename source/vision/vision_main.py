import argparse

def create_parser():
    """ Create the main parser for the Vision Pkg

    Returns
    -------
        args : argparse 'parser' object
            contains all user arguments
    """

    parser = argparse.ArgumentParser(description='Control Vision')

    parser.add_argument('--train',
                        dest = 'train',
                        action = 'store_true',
                        help = "Start S"