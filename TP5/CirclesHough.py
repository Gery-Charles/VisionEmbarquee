import sys
import cv2 as cv
import numpy as np

# --- Configuration Camera ---
cam = 0
cap = cv.VideoCapture(cam)
cap.set(cv.CAP_PROP_FPS, 25)

# --- Chargement de la calibration ---
def loadCameraCalibration():
    cpf = "charuco_calibration.npz"
    try:
        with np.load(cpf) as data:
            return data['camera_matrix'], data['dist_coeff']
    except Exception as e:
        print(f"Erreur : {e}")
        return None, None

intrinsic, distortion = loadCameraCalibration()

if intrinsic is None:
    sys.exit("Calibration introuvable.")

ret, first_frame = cap.read()
h, w = first_frame.shape[:2]

# alpha = 0 : on zoome pour retirer les bords noirs
# alpha = 1 : on garde tous les pixels, même avec des zones noires
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(intrinsic, distortion, (w, h), alpha=0)

# On recalcule les maps avec cette NOUVELLE matrice
map1, map2 = cv.initUndistortRectifyMap(
    cameraMatrix=intrinsic, 
    distCoeffs=distortion, 
    R=None, 
    newCameraMatrix=new_camera_matrix, 
    size=(w, h), 
    m1type=cv.CV_32FC1
)

# --- Traitement de la vidéo ---
while True:
    ret, frame_raw = cap.read()
    if not ret:
        break

    # Application de la calibration
    inimg = cv.remap(frame_raw, map1, map2, interpolation=cv.INTER_LINEAR)

    # Détection de couleur
    hsv = cv.cvtColor(inimg, cv.COLOR_BGR2HSV)
    jaune_bas = np.array([20, 80, 80])
    jaune_haut = np.array([40, 255, 255])
    mask = cv.inRange(hsv, jaune_bas, jaune_haut)
    jaune_mask = cv.medianBlur(mask, 5)

    # Détection de cercles
    rows = jaune_mask.shape[0]
    circles = cv.HoughCircles(jaune_mask, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=18,
                               minRadius=5, maxRadius=100)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # calcul de la distance à partir du rayon (en pixels) et de la taille réelle du cercle (en mm)
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            focale = (new_camera_matrix[0, 0] + new_camera_matrix[1, 1]) / 2
            R_reel = 60  # taille réelle du cercle en mm

            # Dessin centre et contour
            cv.circle(inimg, center, 1, (0, 255, 0), 3)
            cv.circle(inimg, center, radius, (255, 0, 255), 3)

            # calcul de la distance du cercle en cm
            # Thalès
            distance = ((focale * R_reel) / radius) / 10  # conversion en cm

            cv.putText(inimg, f"Dist: {distance:.1f} cm", (center[0]-20, center[1]-20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv.imshow("frame calibrated", inimg)
    cv.imshow("Masque Jaune", jaune_mask)

    if cv.waitKey(1) == 27: # ESC
        break

cap.release()
cv.destroyAllWindows()