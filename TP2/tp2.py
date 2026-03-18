import cv2 as cv
import numpy as np

image1 = 'taxi/taxi01.pgm' 
image2 = 'taxi/taxi02.pgm' 

frame1 = cv.imread(image1)
frame2 = cv.imread(image2)

cv.imshow('Image 1', frame1)
cv.imshow('Image 2', frame2)
cv.waitKey(0)


# Conversion en niveaux de gris
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

# Calcul du Dense Optical Flow (Farneback algo)
flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)


# Convertion des coordonnées cartésiennes en polaires
# mag : norme (vitesse du déplacement)
# ang : angle (direction)
mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

cv.imshow('Norme Brute', mag)
cv.waitKey(0)


seuil = 0.75
seuil2 = 2 

# On garde la norme si > 0.75, sinon 0
mag_seuillee = np.where(mag > seuil, mag, 0)

cv.imshow(f'Seuil={seuil}', mag_seuillee)
cv.waitKey(0)

#carte binaire des pixels en déplacement
res, mag_binaire = cv.threshold(mag, seuil, 255, cv.THRESH_BINARY)
# ou 
# mag_binaire = cv.threshold(mag, seuil, 255, cv.THRESH_BINARY)[1]
cv.imshow(f'Seuil binaire={seuil}', mag_binaire)
cv.waitKey(0)



cv.destroyAllWindows()