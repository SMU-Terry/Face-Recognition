import cv2 as cv
import numpy as np
import dlib
import matplotlib.pyplot as plt

cap = cv.VideoCapture(0)

# face_detector = cv.CascadeClassifier('./face_detection/face_detection/cascades/haarcascade_frontalface_default.xml')
face_detector = dlib.get_frontal_face_detector()

while True:
    ret, frame = cap.read()

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # detections = face_detector.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=7)
    detections = face_detector(frame, 1)
    for face in detections:
        l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
        cv.rectangle(frame, (l,t), (r,b), (0,255,0), 2)

    frame = cv.flip(frame, 1)

    cv.imshow('Demo', frame)
    
    if cv.waitKey(10) & 0xff==27:
        break

cap.release()
cv.destroyAllWindows()
