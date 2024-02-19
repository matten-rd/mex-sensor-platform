import cv2

from YOLOv8_Tracker import YOLOv8_Tracker

def main():
    tracker = YOLOv8_Tracker(
        classes=0,
        modelName='yolov8n.pt'
    )

    if (tracker.cap.isOpened()):

        while True:
            frame = tracker.trackObjects()
           
            tracker.showVideoFeed(frame)
            tracker.captureVideoFeed(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break   
            
        tracker.destroyVideoFeed()

if __name__ == "__main__":
    main()