"""
The `train_centernet_hourglass.py` script is intented to train EfficientNetD5
models via the TensorFlow 2 object detection api

EfficientNet paper: https://arxiv.org/abs/1911.09070
"""

import argparse

def init_parser():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    settings:dict = init_parser()

if __name__ == '__main__':
    main()
