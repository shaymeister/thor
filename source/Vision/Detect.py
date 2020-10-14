import cv2
import numpy as np
import os

import tensorflow as tf

from mtcnn import MTCNN

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils

class Detect:
    """
    TODO Finish Documentation (Numpy Style)
    """
    
    # class attributes
    model = None
    category_index = None
    detector = None

    def __init__(
            self, 
            model_path = "models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model",
            label_path = "models/mscoco_complete_label_map.pbtxt"):
        """
        TODO Finish Documentation (Numpy Style)
        """

        # load model
        try:
            self.model = tf.saved_model.load(model_path)
        except FileNotFoundError:
            print("Unable to load model from {}".format(model_path))
        
        # load label map data
        try:
            self.category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)
        except FileNotFoundError:
            print("Unable to load labels from {}".format(label_path))

    def inference(self, img, detect_faces=True):
        """
        TODO Finish Documentation (Numpy Style)
        """

        # convert the rgb_img to tensor for model and add it to batch
        img_tensor = tf.convert_to_tensor(img)
        img_tensor = img_tensor[tf.newaxis]

        # inference argued image
        detections = self.model(img_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be integers
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # copy original image
        image_np_with_detections = img.copy()

        # extract desired information from detections dictionary
        detection_boxes = detections['detection_boxes']
        detection_classes = detections['detection_classes']
        detection_scores = detections['detection_scores']

        # create empty attributes to hold desired class
        boxes = []
        classes = []
        scores = []

        # loop through all detections
        for i in range(len(detection_scores)):
            # look for human objects
            if detection_classes[i] == 1: # 1 is human
                boxes.append(detection_boxes[i])
                classes.append(detection_classes[i])
                scores.append(detection_scores[i])
            else:
                continue
        
        # convert lists to numpy arrays
        boxes = np.array(boxes)
        classes = np.array(classes)
        scores = np.array(scores)

        if detect_faces:
            image_np_with_detections = self.detect_faces(
                                            image=image_np_with_detections,
                                            boxes=boxes,
                                            classes=classes,
                                            scores=scores,
                                            confidence_threshold=0.3)

        # show detections on copied image
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes,
            classes,
            scores,
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        # return inference
        return image_np_with_detections

    def detect_faces(self, image, boxes, classes, scores, confidence_threshold):
        """
        TODO Finish Documentation (Numpy Style)
        """
        
        # TODO Make sure human object is in frame

        if self.detector is None:
            self.detector = MTCNN()

        # detect faces
        detections = self.detector.detect_faces(image)

        # loop through all face detections
        for i in range(len(detections)):
            # convert detections to array
            face_boxes = detections[i]['box']
            face_keypoints = detections[i]['keypoints']
            face_confidence = detections[i]['confidence']
            
            # make sure confidence is above threshold
            if face_confidence < confidence_threshold:
                continue

            # draw bounding box
            image = cv2.rectangle(image,
                    (face_boxes[0], face_boxes[1]),
                    (face_boxes[0] + face_boxes[2], face_boxes[1] + face_boxes[3]),
                    (0,155,255),
                    2)

            # draw keypoints
            cv2.circle(image,(face_keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(image,(face_keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(image,(face_keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(image,(face_keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(image,(face_keypoints['mouth_right']), 2, (0,155,255), 2)

        # return image
        return image.copy()
