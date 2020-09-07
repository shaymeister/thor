# import required libraries
from vidgear.gears import NetGear
from datetime import datetime
import cv2

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = cv2.VideoCapture(0) 

# define tweak flags
options = {'flag' : 0, 'copy' : False, 'track' : False}

# Define Netgear Client at given IP address and define parameters (!!! change following IP address '192.168.x.xxx' with yours !!!)
server = NetGear(address = '192.168.1.175', port = '5454', protocol = 'tcp',  pattern = 0, logging = True, **options)

# define writer
CODEC = 'XVID'
fourcc = cv2.VideoWriter_fourcc(*CODEC)
FPS = 30
RES = (640, 480)
date = datetime.now()
out = cv2.VideoWriter('videos/recording_'
    + str(date.month) + '-'
    + str(date.day) + '-'
    + str(date.year) + '_'
    + str(date.hour) + '-'
    + str(date.minute) + '-'
    + str(date.second) + '_'
    + '.avi', fourcc, FPS, RES, True)

# loop over until KeyBoard Interrupted
while True:

  try: 

    # read frames from stream
    (grabbed, frame) = stream.read()

    # check for frame if not grabbed
    if not grabbed:
      break

    print(frame.shape)

    # save the frame
    out.write(frame)

    # send frame to server
    server.send(frame)

  except KeyboardInterrupt:
    break

# safely close video stream
stream.release()

# safely close server
server.close()

out.release()