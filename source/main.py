from Config import Config
import Vision

def main(config):
    """starting node for Thor"""

    if config.getStartVision():
        start_vision(args)


# Determine if vision_main.py is being executed
# directly or from another script
if __name__ == "__main__":
    # create Config object
    config = Config()

    # create cmd-line parser
    config.create_argparser()

    # start
    main(config)
