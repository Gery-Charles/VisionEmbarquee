import libjevois as jevois
import cv2
import numpy as np

class Camshift:
    def __init__(self):
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
        self.track_mode = False
        self.termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.roiBox = (130, 90, 60, 60) 
        self.roiHist = None

    def parseSerial(self, msg):
        # .strip() est vital pour nettoyer les caractères invisibles (\r, \n)
        msg = msg.strip()
        jevois.LINFO("Command: [{}]".format(msg))
        
        if msg == "off":
            self.track_mode = False
            return "Tracking OFF"
        if msg == "on":
            # On force le recalcul de l'histogramme au prochain cycle
            self.track_mode = False 
            return "Tracking RESET & ON"
        return "ERR: Unknown command"

    def process(self, inframe, outframe):
        frame = inframe.getCvBGR()
        self.timer.start()
        
        # Prétraitement
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Masque pour isoler la couleur (élimine le bruit blanc/noir)
        mask = cv2.inRange(hsv, np.array((0, 15, 0)), np.array((17, 170, 255)))

        if not self.track_mode:
            # --- INITIALISATION ---
            x, y, w, h = self.roiBox
            hsv_roi = hsv[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w]
            
            # Calcul de l'histogramme sur le canal Hue (0) avec le masque
            self.roiHist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
            cv2.normalize(self.roiHist, self.roiHist, 0, 255, cv2.NORM_MINMAX)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.track_mode = True 

        else:
            # --- TRACKING ---
            # Correction : on utilise 'hsv' (image complète), pas 'hsv_roi'
            backProj = cv2.calcBackProject([hsv], [0], self.roiHist, [0, 180], 1)
            
            # On filtre la rétroprojection avec le masque pour plus de stabilité
            backProj &= mask

            # Amélioration morphologique pour nettoyer les points isolés
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            backProj = cv2.morphologyEx(backProj, cv2.MORPH_OPEN, kernel)

            # Application de CamShift
            ret, self.roiBox = cv2.CamShift(backProj, self.roiBox, self.termination) 

            # Dessin de la box rotative
            pts = np.intp(cv2.boxPoints(ret))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
        outframe.sendCv(frame)