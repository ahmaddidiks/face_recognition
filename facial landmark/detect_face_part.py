# USAGE 
# python detect_face_part.py --image images\didik.jpg

import imutils
from imutils import face_utils
import cv2 as cv
import dlib
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", type=str,
            default="shape_predictor_68_face_landmarks.dat",
            help="path to facial landmark predictor")
ap.add_argument("-i", "--image", type=str,
            default="images\didik.jpg",
            help="path to input image")
args = vars(ap.parse_args())

# initialize dblib's face detector (HOSg-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load input image, resize it, and convert it to grayscale
image = cv.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#detect face in grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmark for the face region, then convert landmark (x,y)-coordinates to a NumPy array to numpy arrya
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the face parts individually
    for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        # clone the original and edit the clone image
        clone = image.copy()
        cv.putText(clone, name, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
        
        # loop over the subset of facial landmark, draw the specific part
        for (x,y) in shape[i:j]:
            cv.circle(clone, (x,y), 1, (0, 0, 255), -1)
        
        # extract the ROI of the face region a separate image
        x, y, w, h = cv.boundingRect(np.array([shape[i:j]]))
        roi = image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv.INTER_CUBIC)

        # show the particular face part
        cv.imshow("ROI", roi)
        cv.imshow("Image", clone)
        cv.waitKey(0)

# cisual all facial landmark with a transparant overlay
output = face_utils.visualize_facial_landmarks(image, shape)
cv.imshow("Image", output)
cv.waitKey(0)
