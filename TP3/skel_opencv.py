import cv2
import numpy as np


# Test different values 1,2,3 ...
cam=0 # capture from camera at location 0 

cap = cv2.VideoCapture(cam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
#cap.set(cv2.CAP_PROP_GAIN, 1000)
#cap.set(cv2.CAP_PROP_EXPOSURE, 10)


while True:
    ret, img = cap.read()

    cv2.imshow("input", img)
    key = cv2.waitKey(1)
    if key == 27:
        break


cv2.destroyAllWindows()
cv2.VideoCapture(cam).release()
