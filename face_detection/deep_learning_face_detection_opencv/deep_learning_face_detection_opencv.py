#USAGE
#python deep_learning_face_detection_opencv.py --image images\didik.jpg

#import the necassary packages
import numpy as np
import cv2 as cv
import imutils
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototext", type=str,
            default="deploy.prototxt.txt",
            help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", type=str,
            default="res10_300x300_ssd_iter_140000.caffemodel",
            help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
            help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#load serialized model
print("[INFO] loadign model...")
net = cv.dnn.readNetFromCaffe(args["prototext"], args["model"])

#load the input image and construct an input blob for the image
#by resizing to a fixed 300x300 pixels and then normalizing it'''
image = cv.imread(args["image"])
image = imutils.resize(image, width=600, height=600)
h, w  = image.shape[:2]
#blob = cv.dnn.blobFromImage(cv.resize(image, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
blob  = cv.dnn.blobFromImage(imutils.resize(image, width=300, height=300), 1.0, (300,300), (104.0, 177.0, 123.0))
# pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    #extract the confidence
    confidence = detections[0, 0, i, 2]

    #filter out weak detections by ensuring the `confience` is greater than the minimum confience
    if confidence > args["confidence"]:
        # compute the (x,y)-coordinates box for object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype("int")

        # draw the bounding of the face with probability
        text = f"{(confidence * 100):.2f}"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv.putText(image, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# show the output image
cv.imshow("Output", image)
cv.waitKey(0)
