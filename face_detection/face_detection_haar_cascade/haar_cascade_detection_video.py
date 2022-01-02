#USAGE 
#python haar_cascade_detection_video.py

#import the necassary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2 as cv

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str,
    default="haarcascade_frontalface_default.xml",
    help="path to haar cascade face detector")
args = vars(ap.parse_args())

# load the haar cascade face detector from disk
print("[INFO] loading face detector...")
detector = cv.CascadeClassifier(args["cascade"])

# initialize the video
print("[INFO] starting videostream")
vs = VideoStream(src=0).start()
time.sleep(2)

#loop over the frames from the video stream
while True:
    #grab the frame from the video stream and resize it and convert into grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #perform face detection
    rects = detector.detectMultiScale(gray, scaleFactor=1.05,
        minNeighbors=5, minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE)
    
    #perform over the bounding boxes
    for (x,y,w,h) in rects:
        #draw the face bounding box on the image
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    # show the output
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF

    #if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv.destroyAllWindows()
vs.stop()