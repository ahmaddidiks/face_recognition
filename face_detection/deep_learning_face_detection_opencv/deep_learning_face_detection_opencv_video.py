# USAGE
# python deep_learning_face_detection_opencv_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

#import the necassary packages
import cv2 as cv
import imutils
from imutils.video import VideoStream
import argparse
import time
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototext", type=str,
            default="deploy.prototxt.txt", help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", type=str,
            default="res10_300x300_ssd_iter_140000.caffemodel", help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
            help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load serialized model
print("[INFO] loading model...")
net = cv.dnn.readNetFromCaffe(args["prototext"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    # grab the frame from video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=600, height=600)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence
        confidence = detections[0, 0, i, 2]

        # filter weak detections
        if confidence < args["confidence"]:
            continue
        # compute the (x, y)-coordinates of the bounding box for the
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype("int")

        # draw the bounding box
        text = f"{confidence * 100:.2f}"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv.putText(frame, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output frame
    cv.imshow("Frame", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# clean up
cv.destroyAllWindows()
vs.stop()