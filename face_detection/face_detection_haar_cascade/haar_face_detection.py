#USAGE 
# python haar_face_detection.py --image images\didik.jpg

#import the necessary packages
import argparse
import imutils
import cv2 as cv

# construct the argument parser and paarse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
ap.add_argument("-c", "--cascade", type=str, default="haarcascade_frontalface_default.xml", help="path to haar cascade face detector")
args = vars(ap.parse_args())

#load the haar cascade face detector from
print("[INFO] loading face detector")
detector = cv.CascadeClassifier(args["cascade"])

# load the input image, resize and convert it into grayscale
image = cv.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#detect faces in the input image using the haar cascade face detector
print("[INFO] performing face detection...")
rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
print(f"[INFO] {len(rects)} face detected")

# lop over the bounding boxes
for (x,y,w,h) in rects:
    #draw the face bounding box on the image
    cv.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

#show the output image
cv.imshow("Image", image)
cv.waitKey(0)


