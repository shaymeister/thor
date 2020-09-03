import cv2
from datetime import datetime
import numpy as np

from .Detect import Detect

class Camera:
    """Manage and control the USB camera"""
    cam_num = 1 # 1 is the usb cam on thor

    def __init__(self):
        print("Vision: Initialized Camera")

    def setCamNum(self, int):
        """Change the cam_num attribute to the argued value"""
        self.cam_num = int

    def show_view(self):
        """Show the current camera's view"""

        # define video source
        stream = cv2.VideoCapture(0)

        # check if the video stream is able to be accessed
        if (stream.isOpened()):
            print("Starting Camera.")
        else:
            print("Unable to access camera.")

        # start the streaming loop
        while(stream.isOpened()):
            # capture frame by frame
            ret, frame = stream.read()

            # check if frame is captured correctly
            if not ret:
                print("Unable to receive frame.")

            # display the frame
            cv2.imshow("current view", frame)

            # run until key press 'q'
            if cv2.waitKey(1) == ord('q'):
                break

        # release the capture
        stream.release()
        cv2.destroyAllWindows()

    def record(self, show_view):
        """Record via the usb camera"""

        # define video source
        stream = cv2.VideoCapture(self.cam_num)

        # check if the video stream is able to be accessed
        if (stream.isOpened()):
            print("Starting Camera.")

        # define codec and create VideoWriter
        CODEC = 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        FPS = 30
        RES = (640, 480)
        date = datetime.now()
        out = cv2.VideoWriter('videos/recording_'
                                + str(date.month) + '-'
                                + str(date.day) + '-'
                                + str(date.year) + '_'
                                + date.strftime('%X')
                                + '.avi', fourcc, FPS, RES, True)

        # start the streaming loop
        while(stream.isOpened()):
            # capture frame by frame
            ret, frame = stream.read()

            # make sure the frames are reading
            if not ret:
                print("Unable to receive frame.")
                break

            # write the frame
            out.write(frame)

            # show frame
            if show_view:
                cv2.imshow("Current View", frame)

            # run until key press 'q'
            QUIT_KEY = 'q'
            if cv2.waitKey(1) == ord(QUIT_KEY):
                break

        # release the capture
        stream.release()
        out.release()
        cv2.destroyAllWindows()

    def detect(self):
        
        # define video source
        stream = cv2.VideoCapture(self.cam_num)

        # check if the video stream is able to be accessed
        if (stream.isOpened()):
            print("Starting Camera.")

        # define codec and create VideoWriter
        CODEC = 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        FPS = 30
        RES = (640, 480)
        TENSOR_RES = (800, 600)
        date = datetime.now()
        out = cv2.VideoWriter('videos/recording_'
                                + str(date.month) + '-'
                                + str(date.day) + '-'
                                + str(date.year) + '_'
                                + date.strftime('%X')
                                + '.avi', fourcc, FPS, RES)
        out2 = cv2.VideoWriter('videos/recording_detect_'
                        + str(date.month) + '-'
                        + str(date.day) + '-'
                        + str(date.year) + '_'
                        + date.strftime('%X')
                        + '.avi', fourcc, FPS, TENSOR_RES)

        # initialize detector
        detector = Detect()
        detector.start()

        # start the streaming loop
        while(stream.isOpened()):
            # capture frame by frame
            ret, frame = stream.read()

            # make sure the frames are reading
            if not ret:
                print("Unable to receive frame.")
                break

            # write the frame
            out.write(frame)

            # send the frame through the object detector
            frame = detector.detect(frame)

            # write the modified frame
            out2.write(frame)

            # show frame
            # TODO Update Following If-statement
            if 1 == 1:
                cv2.imshow("God's Eye", frame)

            # run until key press 'q'
            QUIT_KEY = 'q'
            if cv2.waitKey(1) == ord(QUIT_KEY):
                break

        # release the capture
        stream.release()
        out.release()
        out2.release()
        cv2.destroyAllWindows()


