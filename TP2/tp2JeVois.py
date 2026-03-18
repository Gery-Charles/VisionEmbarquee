import libjevois as jevois
import cv2 as cv
import numpy as np

class OpticalFlow:
    def __init__(self):
        self.old_gray = None
        self.timer = jevois.Timer('FlowGray', 50, jevois.LOG_DEBUG)

        self.moving_pixels = 0.0

    def process(self, inframe, outframe):

        # Acquisition et conversion
        frame_bgr = inframe.getCvBGR()
        frame_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        self.timer.start()
        
        # Initialisation au premier frame
        if self.old_gray is None:
            self.old_gray = frame_gray.copy()
            outframe.sendCv(frame_bgr)
            return

        # Calcul du flux Farneback (Paramètres ajustés pour la performance)
        flow = cv.calcOpticalFlowFarneback(self.old_gray, frame_gray, None, 
                                          0.5, 3, 15, 3, 5, 1.2, 0)

        # Calcul de la magnitude (vitesse)
        mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalisation et Seuil
        # Le flux Farneback renvoie des valeurs réelles. 
        # Pour l'affichage, on multiplie souvent par un facteur pour voir le mouvement.
        seuil = 0.75
        _, mag_thresh = cv.threshold(mag, seuil, 255, cv.THRESH_BINARY)
        
        # Conversion en 8-bit pour l'affichage
        display_frame = mag_thresh.astype(np.uint8)

        # Mise à jour de l'état
        self.old_gray = frame_gray.copy()

        display_bgr = cv.cvtColor(display_frame, cv.COLOR_GRAY2BGR)

        # calcul du pourcentage de pixels en mouvement
        self.moving_pixels = 100.0 * np.count_nonzero(mag_thresh) / mag_thresh.size
        cv.putText(display_frame, f'Moving Pixels: {self.moving_pixels:.2f}%', 
                   (3, 30), cv.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        
        if self.moving_pixels > 30.0:
            cv.putText(display_bgr, "Detected ! ", (100, 230), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Affichage des FPS
        fps = self.timer.stop()
        cv.putText(display_frame, fps, (3, 15), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)

        # Envoi (On convertit en BGR pour assurer la compatibilité avec l'outframe)
        outframe.sendCv(display_bgr)