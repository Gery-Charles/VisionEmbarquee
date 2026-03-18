import libjevois as jevois
import cv2 as cv
import numpy as np
import math

class testJevois:
    def __init__(self):
        # Calibration
        w, h = 320, 240
        self.camera_matrix, self.dist_coeff = self.loadCameraCalibration(w, h)
        
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff, (w,h), 1, (w,h))
        self.map1, self.map2 = cv.initUndistortRectifyMap(self.camera_matrix, self.dist_coeff, None, new_camera_matrix, (w, h), cv.CV_32FC1) 

        self.f = (new_camera_matrix[0,0] + new_camera_matrix[1,1]) / 2.0
        self.DIAMETRE_REEL_CM = 4.0
        jevois.LINFO(f"Module initialized. Focal: {self.f:.2f}")

    def loadCameraCalibration(self, w, h):
        cpf = f"/jevois/share/camera/calibration-ov9650-{w}x{h}-standard.yaml"
        fs = cv.FileStorage(cpf, cv.FILE_STORAGE_READ)
        
        if fs.isOpened():
            camMatrix = fs.getNode("camera_matrix").mat().astype(np.float32)
            distCoeffs = fs.getNode("distortion_coefficients").mat().astype(np.float32)
            jevois.LINFO(f"Parameters loaded from {cpf}")
        else:
            jevois.LERROR(f"Failed to load {cpf}. Using defaults!")
            camMatrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
            distCoeffs = np.zeros((5, 1), dtype=np.float32)
            
        return camMatrix, distCoeffs

    # Traitement de l'image
    def process(self, inframe, outframe):
        src = inframe.getCvBGR()
        src = cv.remap(src, self.map1, self.map2, cv.INTER_LINEAR)

        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        mask_jaune = cv.inRange(hsv, (25, 100, 100), (35, 255, 255))
        
        blurred = cv.medianBlur(mask_jaune, 5)
        
        canny = cv.Canny(blurred, 100, 200)
        contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for zone in contours:
            # Filtre pour éliminer les petites cibles ( le bruit )
            area = cv.contourArea(zone)
            if area < 100:
                continue
                
            perimeter = cv.arcLength(zone, True)
            if perimeter > 0:
                q = (4 * math.pi * area) / (perimeter**2)
                
                if q > 0.85:
                    apparent_diameter = 2 * math.sqrt(area / math.pi)
                    distance_cm = (self.DIAMETRE_REEL_CM * self.f) / apparent_diameter
                    
                    jevois.sendSerial(f"DIST: {int(distance_cm)}\n")
                    
                    (x, y), radius = cv.minEnclosingCircle(zone)
                    cv.circle(src, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    cv.putText(src, f"{int(distance_cm)}cm", (int(x), int(y)), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        outframe.sendCv(src)