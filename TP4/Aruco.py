import cv2.aruco as aruco
import cv2
import numpy as np

# print(cv2.__version__)

# --- Configuration Caméra ---
cam = 0
cap = cv2.VideoCapture(cam)
cv2.namedWindow("frame")

w = 640
h = 480

# --- DEFINITIONS DE FONCTIONS ---

# Charger les paramètres de calibration de la caméra
def loadCameraCalibration(w, h):
        cpf = f"calibration{w}x{h}.yaml"
        fs = cv2.FileStorage(cpf, cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            camMatrix = fs.getNode("camera_matrix").mat()
            distCoeffs = fs.getNode("distortion_coefficients").mat()
            print(f"Loaded camera calibration from {cpf}")
        else:
            print(f"Failed to read camera parameters from file [{cpf}]")
        return camMatrix, distCoeffs

# estimation de la pose d'un marqueur à partir de ses coins détectés
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    https://stackoverflow.com/questions/75750177/solve-pnp-or-estimate-pose-single-markerswhich-is-better
    This will estimate the rvec and tvec for each of the marker corners detected by:
    corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
    [marker_size / 2, marker_size / 2, 0],
    [marker_size / 2, -marker_size / 2, 0],
    [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)

    return rvecs, tvecs, trash

def incrustation_image(pts,image_path):
    image = cv2.imread(image_path)
    corners = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])
    mtx_homographique = cv2.getPerspectiveTransform(corners.astype(np.float32), pts.astype(np.float32))
    incrustation = cv2.warpPerspective(image, mtx_homographique, (640, 480))
    return incrustation

intrinsec, distortion = loadCameraCalibration(w, h)

# --- traitement de la vidéo ---
while True:
    ret, inimg = cap.read()

    if not ret:
        break

    dico = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters=aruco.DetectorParameters()

    # detection de marqueurs
    detector = aruco.ArucoDetector(dico,parameters)
    corners, ids, rejectedImgPos = detector.detectMarkers(inimg)

    # dessin des marqueurs détectés
    outimg = aruco.drawDetectedMarkers(inimg, corners, ids)

    # estimation de la pose des marqueurs détectés
    if ids is not None:
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, 0.06, intrinsec, distortion)

        for i in range(len(ids)):
            # dessine les axes du repère de chaque marqueur détecté
            outimg = cv2.drawFrameAxes(outimg, intrinsec, distortion, rvecs[i], tvecs[i], 0.03)
            
            # projette les coordonnées d'un cube dans le repère du marqueur sur l'image
            coordonnees = np.array([[0.03, 0.03, 0], [0.03, -0.03, 0], [-0.03, -0.03, 0], [-0.03, 0.03, 0],
                                    [0.03, 0.03, 0.06], [0.03, -0.03, 0.06], [-0.03, -0.03, 0.06], [-0.03, 0.03, 0.06]], dtype=np.float32)
            
            imagePoints, _ = cv2.projectPoints(coordonnees, rvecs[i], tvecs[i], intrinsec, distortion)

            # imagePoints to int
            imagePoints = np.int32(imagePoints)
            
            # dessine les arêtes du cube projeté sur l'image
            outimg = cv2.drawContours(outimg, [imagePoints[:4]], -1, (0, 255, 0), 2)
            for j in range(4):
                outimg = cv2.line(outimg, tuple(imagePoints[j][0]), tuple(imagePoints[j+4][0]), (255, 0, 0), 2)
                # face superieure du cube
                outimg = cv2.drawContours(outimg, [imagePoints[4:]], -1, (255, 0, 0), 2)

            # incrustation d'une image sur le marqueur détecté
            coordonneesPlan = np.array([[-0.03, 0.03, 0.06], [0.03, 0.03, 0.06], [0.03, -0.03, 0.06], [-0.03, -0.03, 0.06]], dtype=np.float32)
            coordonneesCoinsImg = cv2.projectPoints(coordonneesPlan, rvecs[i], tvecs[i], intrinsec, distortion)[0].reshape(-1, 2)

            incrustation = incrustation_image(coordonneesCoinsImg, "test.png")
            outimg = cv2.addWeighted(outimg, 1, incrustation, 1, 0)

    cv2.imshow("frame", outimg)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()