from .Camera import Camera

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
            cam_num = config.getCamNum(),
            video_path = config.getVideoPath()
        )

        # see if the user wants to detect or simply record
        if config.getVisionDetect(): # user wants to detect video
            # set camera config
            self.camera.configure(
                fps = config.getVisionFPS(),
                image_size = config.getVisionImageSize(),
                record = config.getVisionRecord(),
                show_view = config.getVisionShowView(),
                tensor_image_size = config.getVisionTensorImageSize()
            )

            # start detection
            self.camera.detect()
        elif not config.getVisionDetect(): # user doesn't want to detect video
            # set camera config
            self.camera.configure(
                fps = config.getVisionFPS(),
                image_size = config.getVisionImageSize(),
                record = config.getVisionRecord(),
                show_view = config.getVisionShowView()
            )

            # start video stream
            self.camera.startVideoStream()

            # start recording
            self.camera.record()
        else: # unexpected error
            print("An unexpected error occurred! config.VISION.DETECT \
                   should be a boolean value.")
            sys.exit(0)