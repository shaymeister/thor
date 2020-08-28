import argparse
import Kitt
import Vision

def create_argparser():
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
    kitt_group.add_argument('--kitt',
                            dest   = 'kitt',
                            action = 'store_true',
                            help   = 'Start K.I.T.T')

    vision_group = parser.add_argument_group("vision_group")
    vision_group.add_argument('--vision',
                              dest   = 'vision',
                              action = 'store_true',
                              help   = 'Start Vision')
    vision_group.add_argument('--cam_num',
                              dest = 'cam_num',
                              type = int,
                              default = -1,
                              help = "Set desired camera for vision pkg")
    vision_group.add_argument('--show_cam',
                              dest = 'show_view',
                              action = 'store_true',
                              help = 'Show camera view in GUI')        

    args = parser.parse_args()

    return args

def main():
    """starting node for Thor"""

    args = create_argparser()
    
    if args.kitt:
        start_kitt(args)

    if args.vision:
        start_vision(args)

def start_kitt(args):
    """start kitt and manage functionality"""
    print("Start KITT")

def start_vision(args):
    """start vision and manage functionality"""
    print("Starting Vision")

    cam = Vision.Camera()

    # change cam_num if different from default
    DEFAULT_CAM_NUM = -1
    if args.cam_num != -1:
        cam.setCamNum(args.cam_num)

    cam.record(show_view = args.show_cam)

# Determine if vision_main.py is being executed
# directly or from another script
if __name__ == "__main__":
    main()
