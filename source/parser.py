import argparse

def create_parser():

    parser = argparse.ArgumentParser(description = "How to control Thor.")

    parser.add_argument('--kitt',
                        type=str,
                        help='Start K.I.T.T')
    parser.add_argument('--vision',
                    type=str,
                    help='Start Vision')

    args = parser.parse_args()

    return args