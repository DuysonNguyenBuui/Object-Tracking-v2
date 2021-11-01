from idlelib import history
import cv2
import numpy as np
from tracker import *

# create tracker obj
tracker = EuclideanDistTracker()

# object detection from stable camera
cap = cv2.VideoCapture("video4.mp4")
def rescaleFrame(frame, scale=0.5):
    # image, video, live video
    width_frame = int(frame.shape[1] * scale)
    height_frame = int(frame.shape[0] * scale)
    video_frame = (width_frame, height_frame)
    return cv2.resize(frame, video_frame, interpolation=cv2.INTER_AREA)


# tao bo phat hien doi tuong
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

while True:
    ret, frame = cap.read()
    resize_frame = rescaleFrame(frame)

    height, width, _ = resize_frame.shape
    # extract region of interest
    roi = resize_frame[340: 720, 450: 1000]

    # tao mat na
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # object detector
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # calculate area and remove small element
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # object tracking
    boxes_ids = tracker.update(detections)
    for boxes_id in boxes_ids:
        x, y, w, h, id = boxes_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Tracking", resize_frame)
    cv2.imshow("roi", roi)
    if cv2.waitKey(15) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows
