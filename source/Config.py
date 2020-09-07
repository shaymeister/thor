class Config():
    """
    TODO Finish Documentation (Numpy Style)
    """
    def __init__(self):
        """
        TODO Finish Documentation (Numpy Style)
        """

    def import_config(self):
        """
        TODO Finish Documentation (Numpy Style)
        """

    def create_argparser(self):
        """create command line parser using argparse

        Arguments
        ---------
        TODO Detail Arguments

        Returns
        -------
        args : argparse 'parser' object
            contains all parsing options
        """

        parser = argparse.ArgumentParser(description = "How to control Thor.")

        kitt_group = parser.add_argument_group('kitt_group')
        kitt_group.add_argument('-k', '--kitt',
            dest   = 'kitt',
            action = 'store_true',
            help   = 'Start K.I.T.T')

        vision_group = parser.add_argument_group("vision_group")
        vision_group.add_argument('-v', '--vision',
            dest   = 'vision',
            action = 'store_true',
            help   = 'Start Vision')
        vision_group.add_argument('--cam_num',
            dest    = 'cam_num',
            type    = int,
            default = -1,
            help    = "Set desired camera for vision pkg")
        vision_group.add_argument('--show_cam',
            dest   = 'show_cam',
            action = 'store_true',
            help   = 'Show camera view in GUI')        
        vision_group.add_argument('-t', '-test',
            dest   = 'testing',
            action = 'store_true',
            help   = 'Activate testing mode')

        args = parser.parse_args()
con