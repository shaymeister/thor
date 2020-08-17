import argparse

def create_argparser():
    """ Create the main parser for the Vision Pkg

    Returns
    -------
        args : argparse 'parser' object
            contains all user arguments
    """

    parser = argparse.ArgumentParser(description='Control Vision')

    parser.add_argument('--train',
                        dest = 'train',
                        action = 'store_true',
                        help = "Start S")

def create_regex_parser():
    # TODO Finish Implementation
    print("WARNING: this feature is not currently enabled. To use Vision, "
        + "execute vision_main.py directly from cmd line")


def main(flag):
    if flag:
        create_argparser
    else:
        create_regex_parser

# Determine if vision_main.py is being executed
# directly or from another script
if __name__ == "__main__":
    main(True) # executed directly
else:
    main(False) # executed indirectly
