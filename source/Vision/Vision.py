import os
import re
import sys

from .camera import Camera

class Vision():
    """Manage and Control the Vision Module"""

    # class variables
    config = None # specify script functionality
    camera = None # Camera class object

    def __init__(self, config):
        """Initialize Vision"""

        # save config to class
        self.config = config

    def start(self):
        """Start Vision"""

        if self.config.getVisionVideoPath() is not None:
            self.process_prerecorded()
            return

        # initialize Camera
        self.camera = Camera(cam_num = self.config.getVisionCamNum())

        # see if the user wants to detect or simply record
        if self.config.getVisionDetect(): # user wants to detect video
            # set camera config
            self.camera.configure(
                fps = self.config.getVisionFPS(),
                image_size = self.config.getVisionImageSize(),
                record = self.config.getVisionRecord(),
                show_view = self.config.getVisionShowView(),
                tensor_image_size = self.config.getVisionTensorImageSize())

            # start detection
            self.camera.detect()
        elif not self.config.getVisionDetect(): # user doesn't want to detect video
            # set camera config
            self.camera.configure(
                fps = self.config.getVisionFPS(),
                image_size = self.config.getVisionImageSize(),
                record = self.config.getVisionRecord(),
                show_view = self.config.getVisionShowView())

            # start video stream
            self.camera.startVideoStream()
        else: # unexpected error
            print("An unexpected error occurred! config.VISION.DETECT \
                   should be a boolean value.")
            sys.exit(0)

    def load_data(self):
        """get all video paths from file_path and dir_path"""
        
        # get path
        path = self.config.getVisionVideoPath()
        paths = []

        # check self.file_path
        if path is not None and os.path.isfile(path):
            paths.append(path)
            print('VISION.load_data: found video at {}'.format(path))

        # check self.dir_path
        if path is not None and os.path.isdir(path):
            # find all files in argued directory
            files = os.listdir(path=path)

            # define regex to search for groups in files
            r_groups = re.compile(r'group[0-9]{3}')

            # update user
            print("VISION.load_data: searching for groups of data in {}".format(
                path))

            # filter files in argued directory to groups
            groups = [files for files in files if r_groups.match(files)]
            
            # update user
            print("VISION.load_data: found {} groups of data in {}".format(
                len(groups),
                path))

            # define regex for clips
            r_clips=re.compile(r'recording_.*\.avi')

            # loop through groups
            for group in groups:
                # determine file path to groupXXX
                group_path = os.path.join(path, group, 'clips')
                
                # update user
                print("VISION.load_data - {}: searching for clip(s) in {}".format(
                    group,
                    group_path))

                # get all files from group path
                clips = os.listdir(group_path)

                # filter files in group path based on regex
                clips = [clips for clips in clips if r_clips.match(clips)]

                # update user
                print("VISION.load_data - {}: found {} clip(s) in {}".format(
                    group,
                    len(clips),
                    group_path))

                # add clips to self paths
                for clip in clips:
                    paths.append(os.path.join(group_path, clips[0]))

            # update user on total number of clips
            print('VISION.load_data: found {} video clip(s) in {}'.format(
                len(paths),
                path))

            # end function
            return paths

    def process_prerecorded(self):
        """process all videos in argued directory"""

        # load videos
        clips = self.load_data()

        print(clips)

        # process all loaded clips
        for clip in clips:
            try:
                # update user
                print("VISION: processing {}".format(clip))

                # initialize camera
                self.camera = Camera(video_path=clip)       
                
                # set camera config
                self.camera.configure(
                fps = self.config.getVisionFPS(),
                image_size = self.config.getVisionImageSize(),
                record = self.config.getVisionRecord(),
                show_view = self.config.getVisionShowView(),
                tensor_image_size = self.config.getVisionTensorImageSize())

                # start detection
                self.camera.process_prerecorded()

                # delete camera object
                self.camera = None

                # update user
                print('VISION: processed {}'.format(clip))

                # process next video
                continue
            except KeyboardInterrupt:
                sys.exit()
            #except:
            #    print('VISION: Unable to process {}'.format(clip))
            #    continue

        print("Successfully processed {} clip(s)".format(len(clips)))