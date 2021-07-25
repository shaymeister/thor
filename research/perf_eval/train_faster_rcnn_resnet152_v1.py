"""
The `train_faster_rcnn_resnet152_v1.py` script is intented to train Faster
R-CNN ResNet152v1 models via the TensorFlow 2 object detection api
"""

import argparse

def init_parser():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    settings:dict = init_parser()

if __name__ == '__main__':
    main()
