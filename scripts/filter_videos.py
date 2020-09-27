"""
Skim through the argued videos and delete all frame without movement

TODO Include comments about arguments
"""

import argparse
import cv2
import glob
import imutils
from 
import os
from os import path
from tqdm import tqdm


class VideoFilter():
    """remove RGB frames with no motion from video"""

    def __init__(self):
        """initialize the VideoFIlter class"""
        
        # class attributes
        self.videos = []
        self.valid_codec = ".avi"
        self.blur_region = (21, 21)
        self.format = cv2.VideoWriter_fourcc(*'DIVX')
        self.fps = 24
        self.size = (1920, 1080)

    def load_video(self, video_path):
        """load all frames of video at video into list"""

        # update user
        print("-> Loading {}".format(video_path))

        video = [] # list of all frames in video
        success = True # flag to id end of video stream

        # open video stream
        video_stream = cv2.VideoCapture(video)

        # loop through all frames in video_stream
        while success:
            # get frame
            success, frame = video_stream.read()

            # add frame to video frame list
            video.append(frame)

        # close video stream
        video_stream.close()
        
        # update user
        print("-> Loaded {} frames from {}".format(len(video), video_path))

        # return video frames
        return video

    def load_videos(self, file_path, directory_path):
        """load video(s) from a given path or directory"""

        # if file_path is not equal to None, i.e. there was an argument
        # load video from file_path
        if file_path != None:
            self._get_video(file_path)

        # if dir_path is not equal to None, i.e. there was an argument
        # load all videos from dir_path
        if directory_path != None:
            self._get_videos(directory_path)

    def _get_video(self, file_path):
        """load video from argued path"""

        # get the argued file's extension
        _, file_extension = path.splitext(file_path)

        # if video at file_path doesn't exist, throw error and return
        if not path.exists(file_path):
            print('[ERROR]: {} does not exist!'.format(file_path))
            return

        # if video at file_path isn't a valid format, throw error and return
        if not file_path.endswith(self.valid_codec):
            print("[ERROR]: {} isn't a valid format! {} != {}".format(
                file_path,
                file_extension,
                self.valid_codec)
            )
            return

        # update user
        print('-> Found video: {}'.format(file_path))

        # assuming the video at file_path passed all tests,
        # add said video to self.videos
        self.videos.append(file_path)

    def _get_videos(self, directory_path):
        """load all videos in argued directory"""

        # update user
        print('--> searching for videos in {}'.format(directory_path))

        # get paths to all videos in argued directory
        videos = glob.glob(
            os.path.join(
                directory_path, '*' + self.valid_codec
            )
        )

        # update user
        print('--> found {} videos in {}'.format(len(videos), directory_path))

        # loop through all videos; attempt to import each
        for video in videos:
            self._get_video(video)

    def detect_motion(self, video, min_area):
        """detect motion in video"""

        # update user
        print('-> Looking for frames with motion')

        motion = [] # specify when there is motion in the video

        # get background
        background = video[0]

        # loop over all frames in the video
        for i in tqdm(range(len(video))):
            # get frame at i
            frame = video[i]
            # convert frame to grayscale and blur it
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )
            gray = cv2.GaussianBlur(
                gray,
                self.blur_region,
                0
            )

            # compute the difference between the current
            # and first frame
            frameDelta = cv2.absdiff(background, gray)
            threshold = cv2.threshold(
                frameDelta,
                25,
                255,
                cv2.THRESH_BINARY
            )[1]

            # dilate the threshold to fill in holes,
            # then find contours on threshold image
            threshold = cv2.dilate(
                threshold,
                None,
                iterations=2
            )
            contours = cv2.findContours(
                threshold.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            contours = imutils.grab_contours(contours)

            # loop over the contours
            for contour in contours:
                # check if contour is prominent
                if cv2.contourArea(contour) >= min_area:
                    # set the motion at i to True
                    motion[i] = True
                else:
                    motion[i] = False

        count_true = 0
        count_false = 0
        # count number of frame with and without motion
        for instance in motion:
            if instance:
                count_true += 1
            else:
                count_false += 1

        # update user
        print('-> Found {} frames with motion!'.format(str(count_true)))
        print('-> Found {} frames without motion!!'.format(str(count_false)))
        
        # return motion
        return motion

    def filter_video(self, video_path, save_directory, min_area):
        """filter video to sections with motion"""

        # load video and update user
        video = load_video(video)

        # find motion in video
        motion_timeline = detect_motion(video, min_area)

        # find clips with motion
        clips = find_clips(motion_timeline)

        # export clips to save directory
        export_clips(video, video_path, save_directory, clips)

    def filter_videos(self, save_directory, min_area):
        """filter all loaded videos"""

        # update user
        print('--> Filtering {} Videos'.format(len(self.videos)))

        # loop through all videos in self.videos
        for video in self.videos:
           self.filter_video(video, save_directory, min_area)

        # update user
        print('--> Filtered {} Videos!'.format(len(self.videos)))

    def find_clips(self, motion_timeline):
        """find clips with motion"""

        # update user
        print('-> Looking for video clips with motion')

        clips = [] # list of tuples -> start and end of clips w/ motion

        start_index = None # start index of a given clip
        end_index = None # end index of a given clip

        # loop through motion timeline
        for i in tqdm(range(len(motion_timeline))):
            # get the current instance of motion
            instance = motion_timeline[i]

            # look for start of motion
            if start_index == None and instance == True:
                start_index = i
                continue

            # look for end of motion
            if start_index != None and instance == False:
                end_index = i - 1

                # add clip to clips
                clip = (start_index, end_index)
                clips.append(clip)

                # reset indexes
                start_index = None
                end_index = None
                continue

        # update user
        print('-> Found {} videos clips with motion!'.format(len(clips)))

        # return clips
        return clips

    def export_clips(self, video, video_path, save_directory, clips):
        """export clips with motion to save dir"""

        # get original video base name
        basename = path.basename(video_path)
        
        # update user
        print('--> Exporting {} clips from {}'.format(
            len(clips),
            basename
            )
        )

        # loop through all clips
        for i in tqdm(range(len(clips))):
            # get indexes
            start_index, end_index = clips[i]

            # get frames from indexes
            clip = video[start_index, end_index]

            writer = cv2.VideoWriter(
                path.join(
                    save_directory,
                    'clip{}_{}'.format(
                        i,
                        base_name
                    )
                ),
                self.format,
                self.fps,
                self.dim
            )

            # write all frames in clip to video
            for frame in clip:
                writer.write(frame)

            # close writer
            writer.close

        # update user
        print('--> Finished exporting {} clips from {}'.format(
            len(clips),
            basename
            )
        )

def create_argument_parser():
    """create command-line argument parser"""

    # initialize parser
    parser = argparse.ArgumentParser('Filter Videos')
    
    # create mutually exclusive group where the user
    # will specify the desired video(s)
    locator = parser.add_mutually_exclusive_group(required=True)
    locator.add_argument(
        '-f',
        '--file_path',
        dest='file_path',
        type=str,
        action='store',
        help='path to desired file'
    )
    locator.add_argument(
        '-d',
        '--dir_path',
        dest='directory_path',
        type=str,
        action='store',
        help='path to desired directory'
    )

    # specify save directory of filtered videos
    parser.add_argument(
        '-s',
        '--save_dir',
        dest='save_directory',
        type=str,
        action='store',
        required=False,
        help='directory path to save filtered videos'
    )

    # specify minimum area to classify 'motion' as motion
    parser.add_argument(
        '-a',
        '--min-area',
        dest='min_area',
        type=int,
        default=500,
        required=False,
        help='minimum area size'
    )

    # parser and return parser
    return parser.parse_args()

def main(args):
    """centralize and control all functionality of the script"""

    # initialize the VideoFilter
    video_filter = VideoFilter()

    # load videos
    video_filter.load_videos(
        file_path = args.file_path,
        directory_path = args.directory_path
    )

    # filter videos
    video_filter.filter_videos(
        save_directory = args.save_directory,
        min_area = args.min_area
    )


# check if the script is executed
# directly from the command line
if __name__ == '__main__':
    # since the script is being executed from the command line:
    # create command-line argument parser using 'argpare'
    args = create_argument_parser()

    # call the main method with the user's arguments
    main(args)