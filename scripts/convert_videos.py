"""
Convert video(s) to specified format

TODO: Detail arguments
"""

import argparse
import moviepy.editor as moviepy


class VideoConverter():
    """convert videos"""

    def __init__(self):
        """initialize class variables"""

        self.valid_codec = [
            'mp4',
            'avi',
            'ogv',
            'web']

    def convert_video(self, path, codec):
        """convert video at argued path to argued codec"""

        print('should convert video')
        print(path)
        print(codec)

    def convert_videos(self, path, codec):
        """convert all videos in argued directory to argued codec"""

        print('should convert videos')
        print(path)
        print(codec)


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
        required=True,
        help='desired output format')

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
    vc = VideoConverter()

    # if the user argues '--file_path'
    if args.file_path is not None:
        # convert argued file
        vc.convert_video(args.file_path, args.codec)

    # if the user argues '--dir_path'
    if args.dir_path is not None:
        # convert all images in argued directory
        vc.convert_videos(args.dir_path, args.codec)


# check if script is executed
# directly from CMD line
if __name__ == "__main__":
    main()
