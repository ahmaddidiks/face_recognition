# USAGE
# python facial_landmark.py --image images\didik.jpg

import imutils
from imutils.video import VideoStream
from imutils import face_utils
import dlib
import cv2 as cv
import argparse
import time

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", type=str,
    default="shape_predictor_68_face_landmarks.dat",
    help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv.circle(frame, (x, y), 1, (0, 0, 255), -2)

    # show the output
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# clean up
cv.destroyAllWindows()
vs.stop()
print(type(rects))
print(rects)
