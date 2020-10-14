"""
use the Vision module to detect prerecorded videos
"""

import argparse
import os
import sys

# add source directory to path (contains Vision)
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(parentdir, 'source'))

from Vision import Vision

def create_argparser():
    """create comand line argument parser using argparse"""

    # initialize parser
    parser = argparse.ArgumentParser("Detect prerecorded videos")

    # parse and return arguments
    return parser.parse_args()

def main(args):
    """
    TODO Finish Documentation (Numpy Style)
    """

    print("success")

# check if script is being executed directly from command line
if __name__ == "__main__":
    # start the main method with command line arguments
    main(create_argparser())