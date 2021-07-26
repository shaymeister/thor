"""
download all models for the performance evaluation paper
"""

import argparse
import os
import tarfile
import wget

def load_parser() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dir',
        type = str,
        default = './models/',
        help = 'path to save directory'
    )
    return vars(parser.parse_args())


def main() -> None:
    # define desired models
    model_url_header:str = 'http://download.tensorflow.org/models/object_detection/tf2/'
    models:dict = {
        'CenterNet_HourGlass104_1024x1024': '20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz',
        'EfficientDet_D4_1024x1024': '20200711/efficientdet_d4_coco17_tpu-32.tar.gz',
        'Faster_R-CNN_Inception_ResNet_V2_1024x1024': '20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz',
        'Faster_R-CNN_ResNet152_V1_1024x1024': '20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz',
        'Mask_R-CNN_Inception_ResNet_V2_1024x1024': '20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz',
        'SSD_ResNet152_V1_FPN_1024x1024_RetinaNet152': '20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
    
    }

    # load settings from parser
    settings:dict = load_parser()
    save_dir:str = settings.get('dir')

    # make save directory
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # download and extract each model file
    for model_name in models.keys():
        model_url:str = models[model_name]
        url:str = model_url_header + model_url
        basename:str = os.path.basename(model_url)
        archive_save_path:str = os.path.join(
            save_dir,
            basename
        )

        print(f'Downloading: {model_name} ...')
        wget.download(url, archive_save_path)
        print('\n')

        print(f'Extracting: {model_name} ...')
        tar_file = tarfile.open(archive_save_path, 'r:gz')
        tar_file.extractall(path=save_dir)
        tar_file.close()
        
        # clean up)
        os.remove(archive_save_path)
        
if __name__ == '__main__':
    main()
