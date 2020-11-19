from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

from mtcnn import MTCNN

class MaskDetector():
    """COVID-19 Face Mask Detector"""

    def __init__(self):
        """initialize MaskDetector class"""

        # class attributes
        self.face_detection_model_path = 'models/face_detection'
        self.mask_detection_model_path = 'models/mask_detector.model'
        self.confidence = 0.5

        # load models
        self._load_face_detection_model()
        self._load_mask_detection_model()

        # initialize variables
        self.face_detector_mtcnn = None

    def _load_face_detection_model(self):
        """load face detection model"""

        # load face detector model architecture
        prototxtPath = os.path.sep.join([self.face_detection_model_path, 
            'deploy.prototxt'])

        # load face detector model weights
        weightsPath = os.path.sep.join([self.face_detection_model_path,
            'res10_300x300_ssd_iter_140000.caffemodel'])

        # load model
        self.face_detector = cv2.dnn.readNet(prototxtPath, weightsPath)

    def _load_mask_detection_model(self):
        """load mask detection model"""
        self.mask_detector = load_model(self.mask_detection_model_path)

    def detect_and_predict_mask_mtcnn(self, frame):
        """detect and predict mask in argued frame"""

        # initialize MTCNN
        if self.face_detector_mtcnn is None:
            self.face_detector_mtcnn = MTCNN()

        # get dimensions from original frame
        (orig_h, orig_w) = frame.shape[:2]

        # detect faces on original image
        detections_orig = self.face_detector_mtcnn.detect_faces(frame)
        
        frame = imutils.resize(frame, width=400)

        # get dimensions from resized frame
        (h, w) = frame.shape[:2]

        # convert image to rgb
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces
        detections = self.face_detector_mtcnn.detect_faces(frame)

        # initialize list of faces, their corresponding locations, and the list
        # of predictions from our mask network
        faces = []
        locs = []
        preds = []

        # loop over all detections
        for i in range(len(detections)):
            # extract the confidence for the given detection
            confidence = detections[i]['confidence']
            face_box = detections[i]['box']

            # if confidence is below threshold, continue
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for
			    # the object
                box = (
                    face_box[0],
                    face_box[1],
                    face_box[0] + face_box[2],
                    face_box[1] + face_box[3]
                )

                (startX, startY, endX, endY) = box
			    
                # ensure the bounding boxes fall within the dimensions of
			    # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
			    # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
			    
                # add the face and bounding boxes to their respective
			    # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
		    # for faster inference we'll make batch predictions on *all*
		    # faces at the same time rather than one-by-one predictions
		    # in the above `for` loop
	        faces = np.array(faces, dtype="float32")
	        preds = self.mask_detector.predict(faces, batch_size=32)

        # loop over the detected face locations and their corresponding
	    # locations
        for (box, pred) in zip(locs, preds):
		    # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
		    
            # determine the class label and color we'll use to draw
		    # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		    
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		
            # display the label and bounding box rectangle on the output
		    # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

        # resize the image to the original size
        frame = cv2.resize(frame, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

        # loop through all face detections
        for i in range(len(detections_orig)):
            # convert detections to array
            face_boxes = detections_orig[i]['box']
            face_keypoints = detections_orig[i]['keypoints']
            face_confidence = detections_orig[i]['confidence']
            
            # make sure confidence is above threshold
            if face_confidence < 0.3:
                continue

            # draw bounding box
            frame = cv2.rectangle(frame,
                    (face_boxes[0], face_boxes[1]),
                    (face_boxes[0] + face_boxes[2], face_boxes[1] + face_boxes[3]),
                    (0,155,255),
                    2)

            # draw keypoints
            cv2.circle(frame,(face_keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(face_keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(face_keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(frame,(face_keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(frame,(face_keypoints['mouth_right']), 2, (0,155,255), 2)

        return frame

    def detect_and_predict_mask(self, frame):
        """detect and predict mask in argued frame"""

        # get dimensions from original frame
        (orig_h, orig_w) = frame.shape[:2]
        
        frame = imutils.resize(frame, width=400)

        # get dimensions from resized frame
        (h, w) = frame.shape[:2]

        # get blob from image
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        # initialize list of faces, their corresponding locations, and the list
        # of predictions from our mask network
        faces = []
        locs = []
        preds = []

        # loop over all detections
        for i in range(0, detections.shape[2]):
            # extract the confidence for the given detection
            confidence = detections[0, 0, i, 2]
            
            # if confidence is below threshold, continue
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for
			    # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
			    
                # ensure the bounding boxes fall within the dimensions of
			    # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
			    # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
			    
                # add the face and bounding boxes to their respective
			    # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
		    # for faster inference we'll make batch predictions on *all*
		    # faces at the same time rather than one-by-one predictions
		    # in the above `for` loop
	        faces = np.array(faces, dtype="float32")
	        preds = self.mask_detector.predict(faces, batch_size=32)

        # loop over the detected face locations and their corresponding
	    # locations
        for (box, pred) in zip(locs, preds):
		    # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
		    
            # determine the class label and color we'll use to draw
		    # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		    
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		
            # display the label and bounding box rectangle on the output
		    # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

        # resize the image to the original size
        frame = cv2.resize(frame, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

        return frame

if __name__ == "__main__":
    md = MaskDetector()
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()

        # show the output frame
        cv2.imshow("Your Eye", frame)
    
        # predict frame
        detect_frame = md.detect_and_predict_mask_mtcnn(frame)

        # show the output frame
        cv2.imshow("God's Eye", detect_frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
