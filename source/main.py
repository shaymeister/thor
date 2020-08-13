import argparse

def create_parser():
    """ Creating main parser

    Summary
    -------
    N/A

    Parameters
    ----------
    N/A

    Returns
    -------
        args : argparse 'parser' object
            contains all parsing options
    """

    parser = argparse.ArgumentParser(description = "How to control Thor.")

    parser.add_argument('--kitt',
                        dest='kitt',
                        action="store_true",
                        help='Start K.I.T.T')
    parser.add_argument('--vision',
                        dest='vision',
                        action="store_true",
                        help='Start Vision')

    args = parser.parse_args()

    return args

def main():
    args = create_parser()

    if args.kitt:
        print("Start KITT")

    if args.vision:
        print("Start Vision")

if __name__ == "__main__":
    main()