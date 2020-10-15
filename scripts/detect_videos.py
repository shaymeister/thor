"""
use the Vision module to detect prerecorded videos
"""

import argparse
import os
import re
import sys

# add source directory to path (contains Vision)
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(parentdir, 'source'))

from Vision import Vision

class VideoDetector():
    """detect for certain objects in videos"""

    # class attribute

    def __init__(self, file_path = None, dir_path = None):
        """initialize class"""
        
        # set class attributes
        self.file_path = file_path
        self.dir_path = dir_path
        self.paths = []

    def load_data(self):
        """get all video paths from file_path and dir_path"""

        # check self.file_path
        if self.file_path is not None and os.path.isfile(self.file_path):
            self.paths.append(self.file_path)
            print('load_data: found video at {}'.format(self.file_path))

        # check self.dir_path
        if self.dir_path is not None and os.path.isdir(self.dir_path):
            # find all groups in dataset
            files = os.listdir(path=self.dir_path)
            r = re.compile(r'group[0-9]{3}')
            groups = [files for files in files if r.match(files)]

            # loop through groups
            for group in groups:
                # get all clips
                clips = os.listdir(os.path.join(self.dir_path, group, 'clips'))
                print(clips)
            
def create_argparser():
    """create comand line argument parser using argparse"""

    # initialize parser
    parser = argparse.ArgumentParser("Detect prerecorded videos")

    # add arguments
    locator = parser.add_mutually_exclusive_group(required=True)
    locator.add_argument(
        '-f',
        '--file_path',
        dest='file_path',
        type=str,
        default=None,
        help='path to video' 
    )
    locator.add_argument(
        '-d',
        '--dir_path',
        dest='dir_path',
        type=str,
        default=None,
        help='path to data directory'
    )

    # parse and return arguments
    return parser.parse_args()

def main(args):
    """manage and control script functionality"""

    # initialize video detector
    detector = VideoDetector(
        file_path = args.file_path,
        dir_path = args.dir_path)

    # load data
    detector.load_data()

# check if script is being executed directly from command line
if __name__ == "__main__":
    # start the main method with command line arguments
    main(create_argparser())