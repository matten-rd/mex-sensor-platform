import cv2
from lepton_control import LeptonControl as Lepton

def main():

    camera = Lepton()
    
    while True:
        frame = camera.updateFrame()

        # Apply colourmap - try COLORMAP_JET if INFERNO doesn't work.
        # You can also try PLASMA or MAGMA
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)

        camera.startStreaming(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
            
    camera.stopStreaming()

if __name__ == "__main__":
    main()

