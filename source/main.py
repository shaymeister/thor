import argparse
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

    parser.add_argument('--kitt',
                        dest   = 'kitt',
                        action = 'store_true',
                        help   = 'Start K.I.T.T')
    parser.add_argument('--vision',
                        dest   = 'vision',
                        action = 'store_true',
                        help   = 'Start Vision')

    args = parser.parse_args()

    return args

def main():
    """starting node for Thor"""

    args = create_argparser()
    
    if args.kitt:
        print("Start KITT")

    if args.vision:
        print("Starting Vision")

        cam = Vision.Camera()

        cam.record(show_view = True)

# Determine if vision_main.py is being executed
# directly or from another script
if __name__ == "__main__":
    main()