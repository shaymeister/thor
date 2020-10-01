import argparse
from yacs.config import CfgNode as CN

class Config():
    """
    TODO Finish Documentation (Numpy Style)
    """

    def __init__(self):
        """
        Initialize all main class variables
        """

        # initialize yacs config node (CfgNode)
        self._CONFIG = CN()

        # initialize main sections of settings with default values
        self._initKittConfig()
        self._initVisionConfig()

    def create_argparser(self):
        """create command line parser using argparse

        Arguments
        ---------
        TODO Detail Arguments

        """

        # initialize argparse parser
        parser = argparse.ArgumentParser(description = "How to control Thor.")

        # add arguments to parser
        parser.add_argument('-c', '--config',
                            help="custom config file path",
                            required=False,
                            type=str,
                            default='source/Config/default.yaml')
        parser.add_argument('opts',
                            help='modify default config using the command-line',
                            default=None,
                            nargs=argparse.REMAINDER)

        # parse args
        args = parser.parse_args()

        # update config with argue yaml file
        self._CONFIG.merge_from_file(args.config)

        # update config with argued options
        self._CONFIG.merge_from_list(args.opts)

        # freeze the config to prevent corruption
        self._CONFIG.freeze()

    def _initKittConfig(self):
        """
        TODO Finish Documentation (Numpy Style)
        """
        
        # create Kitt section in config
        self._CONFIG.KITT = CN()

        # start Kitt module at runtime (True / False)
        self._CONFIG.KITT.START = False

    def _initVisionConfig(self):
        """Initialize configuration for Vision"""

        # create Vision section in config
        self._CONFIG.VISION = CN()

        # start Vision module at runtime (True / False)
        self._CONFIG.VISION.START = False

        # webcam id to receive frames
        self._CONFIG.VISION.CAM_NUMBER = 1 # usb cam on thor

        # start detect package
        self._CONFIG.VISION.DETECT = False

        # frame rate to receive from video source
        self._CONFIG.VISION.FPS = 30

        # image size to receive from video source
        self._CONFIG.VISION.IMAGE_SIZE = [640, 480]

        # record camera footage
        self._CONFIG.VISION.RECORD = False

        # show the camera's view on screen
        self._CONFIG.VISION.SHOW_VIEW = False

        # image size to receive when using detect
        self._CONFIG.VISION.TENSOR_IMAGE_SIZE = [800, 600]

        # video path to receive frames
        self._CONFIG.VISION.VIDEO_PATH = None

    # ---------------
    # getter methods
    # ---------------

    def getKittStart(self):
        """return boolean specifying whether to start Kitt at runtime"""
        return self._CONFIG.KITT.START

    def getVisionCamNum(self):
        """return integer value representing camera source"""
        return self._CONFIG.VISION.CAM_NUMBER

    def getVisionDetect(self):
        """return boolean attribute specifying whether or not to apply object detection"""

    def getVisionFPS(self):
        """return integer value specifying frame rate to record"""
        return self._CONFIG.VISION.FPS

    def getVisionImageSize(self):
        """return list value specifying image size"""
        return self._CONFIG.VISION.IMAGE_SIZE

    def getVisionRecord(self):
        """return boolean value specifying whether to record"""
        return self._CONFIG.VISION.RECORD

    def getVisionShowView(self):
        """return boolean value specifying the initialization of visuals"""
        return self._CONFIG.VISION.SHOW_VIEW

    def getVisionStart(self):
        """return boolean specifying whether to start Vision at runtime"""
        return self._CONFIG.VISION.START

    def getVisionTensorImageSize(self):
        """return boolean specifying output of object detection"""
        return self._CONFIG.VISION.TENSOR_IMAGE_SIZE

    def getVisionVideoPath(self):
        """return path to prerecorded video(s)"""
        return self._CONFIG.VISION.VIDEO_PATH


