import argparse

def create_argparser():
    """ Creating main parser

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

    args = create_argparser()
    
    if args.kitt:
        print("Start KITT")

    if args.vision:
        print("Start Vision")

# Determine if vision_main.py is being executed
# directly or from another script
if __name__ == "__main__":
    main()