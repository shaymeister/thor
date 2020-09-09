import Config
import Kitt
import Vision

def main(config):
    """starting node for Thor"""

    if config.getKittStart():
        # initialize Kitt object
        kitt = Kitt(config)

        # start Kitt
        kitt.start()

    if config.getVisionStart():
        # initialize Vision object
        vision = Vision(config)

        # start Vision
        vision.start()

# Determine if vision_main.py is being executed
# directly or from another script
if __name__ == "__main__":
    # create Config object
    config = Config()

    # create cmd-line parser
    config.create_argparser()

    # start Thor
    main(config)