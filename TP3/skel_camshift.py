import numpy as np
import cv2

roiPts = []
track_mode = False
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
roiBox = None
roiHist = None

def selectROI(event, x, y, flags, param):
    global track_mode, roiPts
    if (event == cv2.EVENT_LBUTTONDOWN) and (len(roiPts) == 4):
        roiPts = []
        track_mode = False
    if (event == cv2.EVENT_LBUTTONDOWN) and (len(roiPts) < 4):
        roiPts.append([x, y])

def getCentroid(img_gray):
    """
    Calcule le centre (xc, yc) d'une image en niveaux de gris (BackProjection)
    selon les formules des moments :
    M00 = somme des intensités
    M10 = somme(x * I)
    M01 = somme(y * I)
    """
    # Création de matrices de coordonnées x et y correspondant à la taille de l'image
    height, width = img_gray.shape
    # x_map contient l'indice de colonne pour chaque pixel, y_map l'indice de ligne
    x_map, y_map = np.meshgrid(np.arange(width), np.arange(height))

    # Formule : M00 = ∑x ∑y I(x, y)
    m00 = np.sum(img_gray)

    # Sécurité pour éviter la division par zéro si l'objet disparaît
    if m00 < 1:
        return None

    # Formule : M10 = ∑x ∑y x * I(x, y)
    m10 = np.sum(x_map * img_gray)
    
    # Formule : M01 = ∑x ∑y y * I(x, y)
    m01 = np.sum(y_map * img_gray)

    # xc = M10/M00 et yc = M01/M00
    xc = int(m10 / m00)
    yc = int(m01 / m00)

    return (xc, yc)

# --- Configuration Caméra ---
cam = 0
cap = cv2.VideoCapture(cam)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", selectROI)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.boxFilter(frame, 0, (5, 5), normalize=True)
    
    # Tracé des points de sélection
    if 0 < len(roiPts) <= 4:
        for x, y in roiPts:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)

    # Initialisation du suivi une fois les 4 points sélectionnés
    if len(roiPts) == 4 and not track_mode:
        roiBox_pts = np.array(roiPts, dtype=np.int32)
        s = roiBox_pts.sum(axis=1)
        tl = roiBox_pts[np.argmin(s)]
        br = roiBox_pts[np.argmax(s)]
        
        roi = frame[tl[1]:br[1], tl[0]:br[0]]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calcul de l'histogramme (Teinte uniquement)
        roiHist = cv2.calcHist([hsv_roi], [0], None, [16], [0, 180])
        cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)

        # Définition de la fenêtre de suivi (x, y, w, h)
        roiBox = (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
        track_mode = True

    # Mode Tracking
    if track_mode:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

        # Nettoyage morphologique
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        backProj = cv2.morphologyEx(backProj, cv2.MORPH_OPEN, disc)
        backProj = cv2.morphologyEx(backProj, cv2.MORPH_CLOSE, disc)

        # Application du CamShift
        r, roiBox = cv2.CamShift(backProj, roiBox, termination)

        pts = np.intp(cv2.boxPoints(r))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # CALCUL DU CENTRE
        x, y, w, h = roiBox
        # les coordonnées restent dans l'image
        y1, y2, x1, x2 = max(0, y), max(0, y+h), max(0, x), max(0, x+w)
        backProj_roi = backProj[y1:y2, x1:x2]

        centroid = getCentroid(backProj_roi)

        if centroid is not None:
            rel_xc, rel_yc = centroid
            # Conversion des coordonnées locales (ROI) en coordonnées globales (Image)
            abs_xc = x1 + rel_xc
            abs_yc = y1 + rel_yc

            # Dessin du centre de masse calculé
            cv2.circle(frame, (abs_xc, abs_yc), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Center: {abs_xc},{abs_yc}", (abs_xc + 10, abs_yc), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("backProj", backProj)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()