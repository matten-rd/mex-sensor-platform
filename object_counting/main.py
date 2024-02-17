import cv2

from YOLOv8_ObjectCounter import YOLOv8_ObjectCounter

def main():
    line_points = [(350, 0), (350, 400)]
    Counter = YOLOv8_ObjectCounter(
        regionPoints=line_points,
        classes=0,
        modelName='yolov8n.pt'
    )

    if (Counter.cap.isOpened()):

        while True:
            frame = Counter.getFullFrame()

            frame = Counter.trackObjects()
           
            Counter.showVideoFeed(frame)
            Counter.captureVideoFeed(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break   
            
        Counter.destroyVideoFeed()

main()