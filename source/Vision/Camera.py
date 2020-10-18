import cv2
from datetime import datetime
import numpy as np
import os
import re
import sys
from tqdm import tqdm

from .detect import Detect

class Camera:
    """Manage and control the USB camera"""

    # -----------------
    # class attributes
    # -----------------
    
    # video source
    cam_num = 1 # 1 is the usb cam on thor
    video_path = None

    # camera configuration
    fps = None
    image_size = None
    record = None
    show_view = None
    tensor_image_size = None

    # streamers and recorder
    stream = None
    video_recorder = None
    video_detect_recorder = None


    def __init__(self, cam_num = 1, video_path = None):
        """Initialize Camera Class"""

        # set class variables
        self.cam_num = cam_num
        self.video_path = video_path

    def configure(self, fps = 30, image_size = [640, 480], record = False,
            show_view = False, tensor_image_size = [800, 600]):
        """configure camera specifications"""
        self.fps = fps
        self.image_size = (image_size[0], image_size[1])
        self.record = record
        self.show_view = show_view
        self.tensor_image_size = tensor_image_size

    def detect(self):
        """
        TODO Finish Documentation Numpy Style
        """

        if self.video_path is None: # stream from camera
            self.stream = cv2.VideoCapture(self.cam_num)
        elif self.video_path is not None: # stream from video
            self.stream = cv2.VideoCapture(self.video_path)
        else: # unexpected error
            print("An unexpected error occurred when starting video stream!")
            sys.exit(0)

        if self.record:
            self.startVideoRecorder()
            self.startVideoDetectionRecorder()

        # initialize detector
        detector = Detect()

        # start the streaming loop
        try:
            while(self.stream.isOpened()):
                # capture frame by frame
                ret, frame = self.stream.read()

                # make sure the frames are reading
                if not ret:
                    print("Unable to receive frame.")
                    break

                # send the frame through the object detector
                detect_frame = detector.inference(frame)

                # check if user wants to record
                if self.record:
                    self.video_recorder.write(frame)
                    self.video_detect_recorder.write(detect_frame)

                # check if user wants to see the cam's view
                if self.show_view:
                    cv2.imshow("Your Eye", frame)
                    cv2.imshow("God's Eye", detect_frame)

                # run until key press 'q'
                QUIT_KEY = 'q'
                if cv2.waitKey(1) == ord(QUIT_KEY):
                    break
        except KeyboardInterrupt:
            pass

        # release the capture
        self.stream.release()
        if self.record:
            self.video_recorder.release()
            self.video_detect_recorder.release()
        cv2.destroyAllWindows()

    def startVideoDetectionRecorder(self, path=None, image_size=None):
        """create recorder for detection video stream"""

        # set attributes for video recorder
        CODEC = 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        date = datetime.now()

        if image_size is not None:
            self.image_size = image_size

        if path is None:
            # create video recorder
            self.video_detect_recorder = cv2.VideoWriter('videos/recording_detect_'
                                + str(date.month) + '-'
                                + str(date.day) + '-'
                                + str(date.year) + '_'
                                + str(date.hour) + '-'
                                + str(date.minute) + '-'
                                + str(date.second) + '_'
                                + '.avi', fourcc, self.fps, self.image_size)
        else:
            # define regex to extract group number
            r_group = re.compile(r'(?<=/)group[0-9]{3}(?=/)')

            # get file name and group from video path
            base_name = "inference_{}".format(os.path.basename(self.video_path))
            group = re.findall(r_group, self.video_path)[0]

            # define output directory
            output_dir = 'output'

            # if output dir does not exist, make it
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            
            # if group in output dir does not exist, make it
            elif not os.path.isdir(os.path.join(output_dir, group)):
                os.mkdir(os.path.join(output_dir, group))
            else:
                pass

            # define video recorder
            self.video_detect_recorder = cv2.VideoWriter(
                                            os.path.join(output_dir,
                                                         group,
                                                         base_name),
                                            fourcc,
                                            self.fps,
                                            self.image_size)

            # end function
            return

    def startVideoRecorder(self):
        """create recorder for video stream"""

        # set attributes for video recorder
        CODEC = 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        date = datetime.now()

        # create video recorder
        self.video_recorder = cv2.VideoWriter('videos/recording_'
                                + str(date.month) + '-'
                                + str(date.day) + '-'
                                + str(date.year) + '_'
                                + str(date.hour) + '-'
                                + str(date.minute) + '-'
                                + str(date.second) + '_'
                                + '.avi', fourcc, self.fps, self.image_size)

    def startVideoStream(self):
        """load video stream from video or camera"""
        
        if self.video_path is None: # stream from camera
            stream = cv2.VideoCapture(self.cam_num)
        elif self.video_path is not None: # stream from video
            stream = cv2.VideoCapture(self.video_path)
        else: # unexpected error
            print("An unexpected error occurred when starting video stream!")
            sys.exit(0)

        # verify stream functionality
        if not stream.isOpened():
            print("ERROR: unable to load video stream.")
            sys.exit(0)

        # set stream to class attribute
        self.stream = stream

        # create video recorder
        if self.record:
            self.startVideoRecorder()

        # start the streaming loop
        try:
            while(self.stream.isOpened()):
                # capture frame by frame
                ret, frame = self.stream.read()

                # make sure the frames are reading
                if not ret:
                    print("Unable to receive frame.")
                    break

                if self.record:
                    self.video_recorder.write(frame)

                # show frame
                if self.show_view:
                    cv2.imshow("Current View", frame)

                # run until key press 'q'
                QUIT_KEY = 'q'
                if cv2.waitKey(1) == ord(QUIT_KEY):
                    break
        except KeyboardInterrupt:
            pass

        # release the capture
        self.stream.release()
        if self.record:
            self.video_recorder.release()
        cv2.destroyAllWindows()

    def process_prerecorded(self):
        """run prerecorded videos through ml pipeline"""

        # start video stream
        self.stream = cv2.VideoCapture(self.video_path)

        # get length of video
        num_frames = self.stream.get(cv2.CAP_PROP_FRAME_COUNT)

        # initialize detector
        detector = Detect()

        # define tqdm object
        progressBar = tqdm(total=num_frames, ncols=100)

        # start the streaming loop
        try:
            while(self.stream.isOpened()):
                # capture frame by frame
                ret, frame = self.stream.read()

                # make sure the frames are reading
                if ret:
                    # check if video recorder has been initialized
                    if self.video_detect_recorder is None:
                        # define video recorder
                        self.startVideoDetectionRecorder(
                            path = self.video_path,
                            image_size = (frame.shape[1], frame.shape[0]))

                    # send the frame through the object detector
                    detect_frame = detector.inference(frame)
                
                    # update progress bar
                    progressBar.update(1)
                
                    # record detected frame
                    self.video_detect_recorder.write(detect_frame)
                elif not ret:
                    break
        except KeyboardInterrupt:
            pass

        # release the capture
        self.stream.release()
        self.video_detect_recorder.release()

        # end function
        return