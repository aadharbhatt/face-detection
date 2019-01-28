import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier(r"C:\Users\Aadhar Bhatt\Anaconda3\Library\etc\haarcascades\haarcascade_frontalface_alt.xml")
cap = cv2.VideoCapture(0)
scaling_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    #face_rects = face_cascade.detectMultiScale(gray,scaleFactor=12, minNeighbors=15,minSize=(20, 20), flags= cv2.CASCADE_SCALE_IMAGE)

    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.imshow('Face Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()