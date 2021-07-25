"""
The `train_efficient.py` script is intented to train EfficientNetD5
models via the TensorFlow 2 object detection api
"""

import argparse

def init_parser():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    settings:dict = init_parser()

if __name__ == '__main__':
    main()
