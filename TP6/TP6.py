from __future__ import print_function
from collections import deque
import cv2 as cv
import numpy as np
import os

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# Load Cascades
face_cascade = cv.CascadeClassifier(get_path('haarcascade_frontalface_default.xml'))
mouth_cascade = cv.CascadeClassifier(get_path('haarcascade_mcs_mouth.xml'))
nose_cascade = cv.CascadeClassifier(get_path('haarcascade_mcs_nose.xml'))
smile_cascade = cv.CascadeClassifier(get_path('haarcascade_smile.xml'))
rpalm = cv.CascadeClassifier(get_path('rpalm.xml'))

pts = deque(maxlen=64)
cap = cv.VideoCapture(0)
mustache_img = cv.imread(get_path('mustache.png'), cv.IMREAD_UNCHANGED)

if face_cascade.empty() or mouth_cascade.empty() or nose_cascade.empty() or smile_cascade.empty():
    print("ERROR: One or more XML files missing from:", BASE_DIR)
    exit()
if mustache_img is None:
    print("ERROR: mustache.png not found in:", BASE_DIR)
    exit()

def draw_track(frame,pts,color=(0,255,0)):
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(pts.maxlen/ (len(pts)-i + 1)) * 2.5)
        cv.line(frame, pts[i - 1], pts[i], color, thickness)

def alpha_mask(frame, img_with_alpha, x, y):
    """ Inserts overlay at (x, y) handling transparency and edges. """
    h_f, w_f = frame.shape[:2]
    h_i, w_i = img_with_alpha.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w_i, w_f), min(y + h_i, h_f)

    # Calculate boundaries for the mustache
    img_x1, img_y1 = max(0, -x), max(0, -y)
    img_x2 = img_x1 + (x2 - x1)
    img_y2 = img_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return frame
    
    overlay = img_with_alpha[img_y1:img_y2, img_x1:img_x2]
    roi = frame[y1:y2, x1:x2]

    # alpha channel (index 3) as the mask
    mask = overlay[:, :, 3] > 0
    roi[mask] = overlay[:, :, :3][mask]

    return frame

while True:
    ret, frame = cap.read()
    if not ret: break

    # Niveaux de gris
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Filtre pour la detection de main
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    faces = face_cascade.detectMultiScale(gray, 1.3, 10)
    rpalms = rpalm.detectMultiScale(blurred, 1.5, 5)

    for (fx, fy, fw, fh) in faces:
        # Face Rectangle
        cv.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
        
        # (ROI)
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        mouth_zone_gray = roi_gray[int(fh * 0.6):fh, :]
        
        # Detect nose inside the face
        noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 10)

        for (nx, ny, nw, nh) in noses:
            # Nose Rectangle
            cv.rectangle(frame, (fx+nx, fy+ny), (fx+nx+nw, fy+ny+nh), (0, 255, 0), 2)

            m_w = int(fw * 0.8)
            m_h = int(mustache_img.shape[0] * (m_w / mustache_img.shape[1]))
            resized_mustache = cv.resize(mustache_img, (m_w, m_h))

            mx = fx + nx + (nw // 2) - (m_w // 2)
            my = fy + ny + nh - (m_h // 2)

            frame = alpha_mask(frame, resized_mustache, mx, my)
            break 

        smiles = smile_cascade.detectMultiScale(mouth_zone_gray, 1.7, 32, minSize=(25, 25))
        if len(smiles) > 0:
            cv.putText(frame, 'SMILE :)', (40, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    # Detection de main (rpalm)
    for (px, py, pw, ph) in rpalms: 
        # Dessin du rectangle autour de la paume
        cv.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
        
        # Detection du centre de la main (= centre du rectangle)
        center = (px + pw // 2, py + ph // 2)

        # Ajout du centre à la file et dessin
        pts.appendleft(center)
        draw_track(frame, pts, color=(0, 0, 255))

    cv.imshow('Detection filters', frame)
    if cv.waitKey(1) == 27 : break

cap.release()
cv.destroyAllWindows()