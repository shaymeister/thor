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

        # initialize Camera
        self.camera = Camera(
            cam_num = self.config.getVisionCamNum(),
            video_path = self.config.getVisionVideoPath()
        )

        # see if the user wants to detect or simply record
        if self.config.getVisionDetect(): # user wants to detect video
            # set camera config
            self.camera.configure(
                fps = self.config.getVisionFPS(),
                image_size = self.config.getVisionImageSize(),
                record = self.config.getVisionRecord(),
                show_view = self.config.getVisionShowView(),
                tensor_image_size = self.config.getVisionTensorImageSize()
            )

            # start detection
            self.camera.detect()
        elif not self.config.getVisionDetect(): # user doesn't want to detect video
            # set camera config
            self.camera.configure(
                fps = self.config.getVisionFPS(),
                image_size = self.config.getVisionImageSize(),
                record = self.config.getVisionRecord(),
                show_view = self.config.getVisionShowView()
            )

            # start video stream
            self.camera.startVideoStream()
        else: # unexpected error
            print("An unexpected error occurred! config.VISION.DETECT \
                   should be a boolean value.")
            sys.exit(0)