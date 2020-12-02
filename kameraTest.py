import numpy as np
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow('pencere')
cv2.moveWindow('pencere', 10, 10)
cv2.namedWindow('gri ekran')
cv2.moveWindow('gri ekran', 690, 10)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height
while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('pencere', frame)
    cv2.imshow('gri ekran', gray)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()