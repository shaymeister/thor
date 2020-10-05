"""
Convert video(s) to specified format

TODO include comments about arguments
"""

import argparse
import glob
import os
import re
import sys
import moviepy.editor as moviepy


class VideoConverter():
    """convert videos"""

    def __init__(self):
        """initialize class variables"""

        # valid codecs
        self.valid_codec = [
            'mp4',
            'avi',
            'ogv',
            'web'
        ]

    def _check_codec(self, codec):
        """check if the argued codec is valid"""

        # loop through all valid codec
        for some_codec in self.valid_codec:
            # see if argued codec matches
            if codec == some_codec:
                return

        # assuming there was no match, throw error and exit
        print("ERROR - Invalid Codec: {}".format(codec))

    @staticmethod
    def _check_path(path, directory):
        """check if the argued file or directory exists"""

        if not os.path.exists(path):
            sys.exit('PATH - {} does not exist!'.format(path))

        if directory and not os.path.isdir(path):
            sys.exit('DIRECTORY - {} does not exist!'.format(path))
        elif not directory and not os.path.isfile(path):
            sys.exit('FILE - {} does not exist!'.format(path))
        elif not directory and os.path.isfile(path) \
                and not re.search(r'\.(mp4|avi|ogv|web)$', path):
            sys.exit('FILE - {} is not a compatible file type'.format(path))
        else:
            return

    def _check_args(self, path, codec, directory):
        """validate argued valued"""

        # validate argued codec
        print('--> checking desired codec: {}'.format(codec))
        self._check_codec(codec)
        print('--> desired codec ({}) is valid'.format(codec))

        # check argued path
        print('--> checking path: {}'.format(path))
        self._check_path(path, directory)
        print('--> path ({}) is valid'.format(path))

    def convert_video(self, path, codec, save_path, check_args):
        """convert video at argued path to argued codec"""

        if check_args:
            # validate args
            print('-> Validating Arguments: codec {}, path {}'.format(codec,
                                                                      path))
            self._check_args(path, codec, directory=False)
            print('-> Validated Arguments: codec {}, path {}'.format(codec,
                                                                     path))

        # load video
        print('-> Loading Video: {}'.format(path))
        vid = moviepy.VideoFileClip(path)
        print('-> Loaded Video: {}'.format(path))

        # convert video
        print('-> Converting Video to {}'.format(codec))
        sliced_path = path[:len(path) - 3]
        new_path = "{}{}".format(os.path.join(
            save_path,
            (os.path.basename(sliced_path))), codec)
        print('New Path: {}'.format(new_path))
        vid.write_videofile(new_path)
        print('-> Converted Video to {} and saving to {}'.format(codec,
                                                                 new_path))

    def convert_videos(self, path, codec, save_path):
        """convert all videos in argued directory to argued codec"""

        # validate args
        self._check_args(path, codec, directory=True)

        # variables
        clips = []
        valid_codecs = ['*mp4', '*avi', '*ogv', '*webm']

        # find all video files in argued path
        for valid_codec in valid_codecs:
            for file in glob.glob(os.path.join(path, valid_codec)):
                clips.append(file)

        # convert all videos to argued codec
        for clip in clips:
            self.convert_video(clip, codec, save_path, check_args=False)


def create_parser():
    """allow user to control script functionality from cmd-line"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--codec',
        dest='codec',
        type=str,
        choices=[
            'mp4',
            'avi',
            'ogv',
            'webm'],
        action='store',
        default='mp4',
        required=False,
        help='desired output format')
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        type=str,
        action='store',
        required=True,
        help='desired save dir')

    locator = parser.add_mutually_exclusive_group(required=True)
    locator.add_argument(
        '--file_path',
        dest='file_path',
        type=str,
        action='store',
        help='path to desired file')
    locator.add_argument(
        '--dir_path',
        dest='dir_path',
        type=str,
        action='store',
        help='path to desired directory')

    return parser.parse_args()


def main():
    """manage and control all functionality"""

    # allow user to interact with script from cmd-line
    args = create_parser()

    # initialize video converter
    video_converter = VideoConverter()

    # if the user argues '--file_path'
    if args.file_path is not None:
        # convert argued file
        video_converter.convert_video(args.file_path,
                                      args.codec,
                                      args.save_path,
                                      check_args=False)

    # if the user argues '--dir_path'
    if args.dir_path is not None:
        # convert all images in argued directory
        video_converter.convert_videos(args.dir_path,
                                       args.codec,
                                       args.save_dir)


# check if script is executed
# directly from CMD line
if __name__ == "__main__":
    main()
