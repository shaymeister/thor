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

    def _initVisionConfig(self):
        """
        TODO Finish Documentation (Numpy Style)
        """


        # create Vision section in config
        self._CONFIG.VISION = CN()

        # start Vision module at runtime (True / False)
        self._CONFIG.VISION.START = False

    def getVisionStart(self):
        """return boolean specifying whether to start Vision at runtime"""
        return self._CONFIG.VISION.START
