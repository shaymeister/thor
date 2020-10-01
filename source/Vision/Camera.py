import cv2
from datetime import datetime
import numpy as np

from .Detect import Detect

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

        print("Vision: Initialized Camera")

    def configure(self, fps = 30, image_size = [640, 480], record = False,
            show_view = False, tensor_image_size = [800, 600]):
        """configure camera specifications"""
        self.fps = fps
        self.image_size = image_size
        self.record = record
        self.show_view = show_view
        self.tensor_image_size = tensor_image_size

    def startVideoStream():
        """load video stream from video or camera"""
        
        if video_path is None: # stream from camera
            stream = cv2.VideoCapture(self.cam_num)
        elif video_path is not None: # stream from video
            stream = cv2.VideoCapture(self.video_path)
        else: # unexpected error
            print("An unexpected error occured when starting video stream!")
            sys.exit(0)

        # verify stream functionality
        if stream.isOpened():
            print("SUCCESS: loaded Video stream")
        else:
            print("ERROR: unable to load video stream.")
            sys.exit(0)

        # set stream to class attribute
        self.stream = stream

    def startVideoRecorder():
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
                                + '.avi', fourcc, self.fps, self.image_size, True)

    def startVideoDetectionRecorder():
        """create recorder for detection video stream"""

        # set attributes for video recorder
        CODEC = 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        date = datetime.now()

        # create video recorder
        self.video_detect_recorder = cv2.VideoWriter('videos/recording_detect_'
                                + str(date.month) + '-'
                                + str(date.day) + '-'
                                + str(date.year) + '_'
                                + str(date.hour) + '-'
                                + str(date.minute) + '-'
                                + str(date.second) + '_'
                                + '.avi', fourcc, self.fps, self.image_size, True)

    def record(self):
        """Record via the usb camera"""

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
            self.video_stream.release()
        cv2.destroyAllWindows()

    def detect(self):

        if self.record:
            startVideoRecorder()
            startVideoDetectionRecorder()

        # initialize detector
        detector = Detect()
        detector.start()

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
                detect_frame = detector.detect(frame)

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
        if record:
            self.video_recorder.release()
            self.video_detect_recorder.release()
        cv2.destroyAllWindows()