import logging
import numpy as np
import cv2
from flirpy.camera.lepton import Lepton
from dataclasses import dataclass

@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

class LeptonControl(Lepton):
    def __init__(self, loglevel=logging.WARNING, roi=ROI(0,0,800,600), viewWidth=800, viewHeight=600):
        super().__init__(loglevel)

        self.roi = roi
        self.viewWidth = viewWidth
        self.viewHeight = viewHeight

    def updateFrame(self):
        raw_img = self.grab().astype(np.float32) # pixel value in centiKelvin
        temp_img = self.centiKelvinToCelsius(raw_img) # pixel value in Celsius

        # Rescale to 8 bit for viewing
        frame = 255*(raw_img - raw_img.min())/(raw_img.max()-raw_img.min())
        frame = frame.astype(np.uint8)

        # resize image
        viewWidth = self.viewWidth
        viewHeight = self.viewHeight
        frame = cv2.resize(frame, (viewWidth, viewHeight), interpolation=cv2.INTER_LANCZOS4)

        # Draw region of interest in frame
        roi = self.roi
        frame = cv2.rectangle(frame, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (255, 32, 32), 3)
        
        # Scale ROI to raw/temp img size and get the pixel matrix from temp_img
        sROI = self.scaleROI(raw_img, viewHeight, viewWidth, roi)
        tempImgROI = temp_img[sROI.y: sROI.y + sROI.h, sROI.x: sROI.x + sROI.w]

        temp_max = np.max(tempImgROI)
        temp_min = np.min(tempImgROI)
        temp_mean = np.mean(tempImgROI)

        frame = self.__displayTemp(frame, temp_min, temp_max, temp_mean)

        return frame

    def scaleROI(self, raw, height, width, roi):
        # Scales ROI from display image to raw/temp image (clip to be sure)
        scale = raw.shape[1] / width
        rx = np.clip(int(roi.x * scale), 0, width - 2)
        ry = np.clip(int(roi.y * scale), 0, height - 2)
        rw = np.clip(int(roi.w * scale), 1, width)
        rh = np.clip(int(roi.h * scale), 1, height)
        return ROI(rx, ry, rw, rh)
    
    def __displayTemp(self, res, min, max, mean):
        text = "MIN: {:.2f} MAX: {:.2f} MEAN: {:.2f}".format(min, max, mean)
        res = self.__putText(res, text, (16, 32))

        return res

    def __putText(self, res, text, point):
        return cv2.putText(res, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
    
    def centiKelvinToCelsius(self, cK):
        return (cK - 27315) / 100
    
    def startStreaming(self, frame):
        cv2.imshow("Lepton", frame)

    def stopStreaming(self):
        self.close()
        cv2.destroyAllWindows()
    
    