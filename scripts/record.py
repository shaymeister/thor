# import required libraries
from datetime import datetime
import cv2

stream = cv2.VideoCapture(0)

if (stream.isOpened()):
            print("Starting Camera.")

        # define codec and create VideoWriter
CODEC = 'XVID'
FPS = 60
RES = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc(*CODEC)
date = datetime.now()
out = cv2.VideoWriter('videos/recording_'
                                + str(date.month) + '-'
                                + str(date.day) + '-'
                                + str(date.year) + '_'
                                + str(date.hour) + '-'
                                + str(date.minute) + '-'
                                + str(date.second)
                                + '.avi', fourcc, FPS, RES, True)
# start the streaming loop
while(stream.isOpened()):
    try:
        # capture frame by frame
        ret, frame = stream.read()

        # make sure the frames are reading
        if not ret:
            print("Unable to receive frame.")
            break

        # write the frame
        out.write(frame)

        # show frame
        if True:
            cv2.imshow("Current View", frame)

        # run until key press 'q'
        cv2.waitKey(1)            

    except KeyboardInterrupt:
        break

stream.release()
out.release()
cv2.destroyAllWindows()