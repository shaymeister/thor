import numpy as np
import cv2

class Camera:
    def __init__(self):
        print("Vision: Initialized Camera")

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
        stream = cv2.VideoCapture(1)

        # check if the video stream is able to be accessed
        if (stream.isOpened()):
            print("Starting Camera.")
        else:
            print("Unable to access camera.")

        # define codec and create VideoWriter
        codec = 'X264'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter('videos/recording.avi', fourcc, 60, (1920, 1080))

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
            if cv2.waitKey(1) == ord('q'):
                break

        # release the capture
        stream.release()
        out.release()
        cv2.destroyAllWindows()
