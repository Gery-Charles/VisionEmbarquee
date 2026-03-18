import sys
import cv2 as cv
import numpy as np

# --- Configuration Caméra ---
cam = 0
cap = cv.VideoCapture(cam)
cap.set(cv.CAP_PROP_FPS, 50)
cv.namedWindow("frame")

# --- traitement de la vidéo ---
while True:
    ret, inimg = cap.read()
    if not ret:
        break

    blur = cv.medianBlur(inimg, 5)
    # seuil cany
    canny = cv.Canny(blur, 100, 200)

    # calcul quotient isopérimétrique
    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            continue
        q = 4 * np.pi * area / (perimeter * perimeter)

        # seuillage sur la circularité
        if q > 0.85 and area > 100:
            # dessin des cercles
            (x, y), radius = cv.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv.circle(inimg, center, radius, (255, 0, 0), 2)

    cv.imshow("frame", inimg)
    cv.imshow("Canny", canny)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()