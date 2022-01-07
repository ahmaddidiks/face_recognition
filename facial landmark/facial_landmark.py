# USAGE
# python facial_landmark.py --image images\didik.jpg

import imutils
from imutils import face_utils
import dlib
import cv2 as cv
import numpy as np
import argparse

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", type=str,
    default="shape_predictor_68_face_landmarks.dat",
    help="path to facial landmark predictor")
ap.add_argument("-i", "--image", type=str,
    default="images/example_01.jpg",
    help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it and convert it to grayscale
image = cv.imread(args["image"])
image = imutils.resize(image, width=600)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
    # determine the faceial landmark for face region,
    # then convert the facial landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
    x, y, w, h = face_utils.rect_to_bb(rect)
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #show the face number
    cv.putText(image, f"Face #{i+1}", (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    for (x,y) in shape:
        cv.circle(image, (x, y), 1, (0, 0, 255), -1)

# show the output
cv.imshow("Output", image)
cv.waitKey(0)