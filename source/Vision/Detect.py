import cv2
import numpy as np
import os
import tarfile
import urllib.request

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import cv2

class Detect:
    """
    TODO Finish Documentation (Numpy Style)
    """
    _MODELS_DIR = None
    _MODEL_DATE = None
    _MODEL_NAME = None
    _MODEL_TAR_FILENAME = None
    MODELS_DOWNLOAD_BASE = None
    _MODEL_DOWNLOAD_LINK = None
    _PATH_TO_MODEL_TAR = None
    _PATH_TO_CKPT = None
    _PATH_TO_CFG = None
    _detection_model = None
    _LABEL_FILENAME = None
    _LABELS_DOWNLOAD_BASE = None
    _PATH_TO_LABELS = None

    def __init__(self):
        """
        TODO Finish Documentation (Numpy Style)
        """
        # self.__downloadModels attributes
        self._MODELS_DIR = 'source/Vision/data/models'
        self._MODEL_DATE = '20200711'
        self._MODEL_NAME = 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8'
        self._MODEL_TAR_FILENAME = self._MODEL_NAME + '.tar.gz'
        self._MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
        self._MODEL_DOWNLOAD_LINK = self._MODELS_DOWNLOAD_BASE + self._MODEL_DATE + '/' + self._MODEL_TAR_FILENAME
        self._PATH_TO_MODEL_TAR = os.path.join(self._MODELS_DIR, self._MODEL_TAR_FILENAME)
        self._PATH_TO_CKPT = os.path.join(self._MODELS_DIR, os.path.join(self._MODEL_NAME, 'checkpoint/'))
        self._PATH_TO_CFG = os.path.join(self._MODELS_DIR, os.path.join(self._MODEL_NAME, 'pipeline.config'))
        self._LABEL_FILENAME = 'mscoco_label_map.pbtxt'
        self._LABELS_DOWNLOAD_BASE = \
            'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
        self._PATH_TO_LABELS = os.path.join(self._MODELS_DIR, os.path.join(self._MODEL_NAME, self._LABEL_FILENAME))

    def _downloadModels(self):
        """
        TODO Finish Documentation (Numpy Style)
        """

        # Download and extract model
        if not os.path.exists(self._PATH_TO_CKPT):
            print('Downloading model. This may take a while... ', end='')
            print(self._MODEL_DOWNLOAD_LINK)
            urllib.request.urlretrieve(self._MODEL_DOWNLOAD_LINK, self._PATH_TO_MODEL_TAR)
            tar_file = tarfile.open(self._PATH_TO_MODEL_TAR)
            tar_file.extractall(self._MODELS_DIR)
            tar_file.close()
            os.remove(self._PATH_TO_MODEL_TAR)
            print('Done')

        # Download labels file
        if not os.path.exists(self._PATH_TO_LABELS):
            print('Downloading label file... ', end='')
            urllib.request.urlretrieve(self._LABELS_DOWNLOAD_BASE + self._LABEL_FILENAME, self._PATH_TO_LABELS)
            print('Done')

    def _loadModel(self):
        """
        TODO Finish Documentation (Numpy Style)
        """

        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(self._PATH_TO_CFG)
        model_config = configs['model']
        self.__detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=self._detection_model)
        ckpt.restore(os.path.join(self.__PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    @tf.function
    def _preprocess(self, image):
        """
        TODO Finish Documentation (Numpy Style)
        """

        image, shapes = self._detection_model.preprocess(image)
        prediction_dict = self._detection_model.predict(image, shapes)
        detections = self._detection_model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    def detect(self, image_np):
        """
        TODO Finish Documentation (Numpy Style)
        """

        category_index = label_map_util.create_category_index_from_labelmap(self.__PATH_TO_LABELS,
                                                                    use_display_name=True)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = self._preprocess(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        return cv2.resize(image_np_with_detections, (800, 600))


    def start(self):
        """
        TODO Finish Documentation (Numpy Style)
        """
        self._downloadModels()
        self._loadModel()