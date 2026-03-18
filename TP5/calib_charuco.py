import cv2
import numpy as np
print(cv2.__version__)

# ==============================
# PARAMÈTRES À ADAPTER
# ==============================

# Nombre de cases du plateau (X, Y)
squaresX = 5
squaresY = 7

# Taille d'une case (mètres)
squareLength = 0.04

# Taille du marqueur ArUco (mètres)
markerLength = 0.02

MIN_IMAGES = 20

# ==============================
# DICTIONNAIRE + PLATEAU
# ==============================

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

board = cv2.aruco.CharucoBoard(
    (squaresX, squaresY),
    squareLength,
    markerLength,
    aruco_dict
)

detector = cv2.aruco.ArucoDetector(aruco_dict)

# ==============================
# STOCKAGE
# ==============================

all_charuco_corners = []
all_charuco_ids = []
image_size = None

# ==============================
# CAPTURE CAMERA
# ==============================

cap = cv2.VideoCapture(0)

print("Montrez le plateau ChArUco sous différents angles.")
print("Appuyez sur ESPACE pour capturer.")
print("Appuyez sur q pour terminer.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        retval, charuco_corners, charuco_ids = \
            cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )

        if retval > 3:
            cv2.aruco.drawDetectedCornersCharuco(
                frame, charuco_corners, charuco_ids
            )

    cv2.imshow("Charuco Calibration", frame)
    key = cv2.waitKey(1)

    if key == ord(' '):  # capture manuelle
        if ids is not None and retval > 3:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            image_size = gray.shape[::-1]
            print(f"Image capturée : {len(all_charuco_corners)}")
        else:
            print("Pas assez de coins détectés.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ==============================
# CALIBRATION
# ==============================

if len(all_charuco_corners) >= 5:

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = \
        cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners,
            all_charuco_ids,
            board,
            image_size,
            None,
            None
        )

    print("\nErreur RMS :", ret)
    print("\nMatrice intrinsèque :\n", camera_matrix)
    print("\nDistorsion :\n", dist_coeffs)

    np.savez("charuco_calibration.npz",
             camera_matrix=camera_matrix,
             dist_coeff=dist_coeffs)

    print("\nCalibration sauvegardée.")

else:
    print("Pas assez d'images valides.")
