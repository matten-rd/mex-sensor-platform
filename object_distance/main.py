from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import math
import os
import configparser
from zed_calibration import *
from utils import *

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')
 
 # object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


def calc_dist(x, y, disparity_scaled, f, B):
    # safely calculate the depth and display it in the terminal
    if (disparity_scaled[y,x] > 0):
        depth = f * (B / disparity_scaled[y,x])
    else:
        depth = 0

    return depth
   

def get_calibration_file(serial_number):
    # serial_number = 23817167
    hidden_path = os.getenv('APPDATA') + '\\Stereolabs\\settings\\'
    calibration_file = hidden_path + 'SN' + str(serial_number) + '.conf'

    if os.path.isfile(calibration_file) == False:
        print('Invalid Calibration File')
        return ""
    
    return calibration_file

def main():
    # get camera calibration
    serial_number = 23817167
    path_to_config_file = get_calibration_file(serial_number)
    if path_to_config_file == "":
        exit(1)

    try:
        # to use a non-buffered camera stream (via a separate thread)
        import camera_stream
        zed_cam = camera_stream.CameraVideoStream()
    except BaseException:
        # if not then just use OpenCV default
        print("INFO: camera_stream class not found - camera input may be buffered")
        zed_cam = cv2.VideoCapture()

    zed_cam.open(0)

    # check open and read video frame
    if zed_cam.isOpened():
        ret, frame = zed_cam.read()
    else:
        print("Error - selected camera not found")
        exit(1)

    height, width, channels = frame.shape

    ################################################################################

    # select config profiles based on image dimensions

    # MODE  FPS     Width x Height  Config File Option
    # 2.2K 	15 	    4416 x 1242     2K
    # 1080p 30 	    3840 x 1080     FHD
    # 720p 	60 	    2560 x 720      HD
    # WVGA 	100 	1344 x 376      VGA

    config_options_width = {4416: "2K", 3840: "FHD", 2560: "HD", 1344: "VGA"}
    config_options_height = {1242: "2K", 1080: "FHD", 720: "HD", 376: "VGA"}

    try:
        camera_mode = config_options_width[width]
    except KeyError:
        print("Error - selected camera : resolution does not match a known ZED configuration profile.")
        exit(1)

    # process config to get camera calibration from calibration file
    # by parsing camera configuration as an INI format file

    cam_calibration = configparser.ConfigParser()
    cam_calibration.read(path_to_config_file)
    fx, fy, B, Kl, Kr, R, T, Q = zed_camera_calibration(cam_calibration, camera_mode, width, height)

    # define display window names
    windowName = "Live Camera Input" # window name
    windowNameD = "Stereo Disparity" # window name

    # set up defaults for stereo disparity calculation
    max_disparity = 128
    window_size = 21

    stereoProcessor = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities = max_disparity, # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=window_size,
            #P1=8 * window_size ** 2,       # 8*number_of_image_channels*SADWindowSize*SADWindowSize
            #P2=32 * window_size ** 2,      # 32*number_of_image_channels*SADWindowSize*SADWindowSize
            #disp12MaxDiff=1,
            #uniquenessRatio=15,
            #speckleWindowSize=0,
            #speckleRange=2,
            #preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_HH
    )

    # set up left to right + right to left left->right + right->left matching +
    # weighted least squares filtering (not used by default)

    left_matcher = stereoProcessor
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_lmbda = 800
    wls_sigma = 1.2

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(wls_lmbda)
    wls_filter.setSigmaColor(wls_sigma)

    # Init Object Counter
    line_points = [(350, 0), (350, 400)]
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=False,
                    reg_pts=line_points,
                    classes_names=model.names,
                    draw_tracks=False)

    if (zed_cam.isOpened()):
        # create window by name (as resizable)
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, width, height)

        cv2.namedWindow(windowNameD, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowNameD, int(width/2), height)

        keep_processing = True

        while keep_processing:
            _, frame = zed_cam.read()

            # split single ZED frame into left an right
            frameL= frame[:,0:int(width/2),:]
            frameR = frame[:,int(width/2):width,:]

            # remember to convert to grayscale (as the disparity matching works on grayscale)
            grayL = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)

            # perform preprocessing - raise to the power, as this subjectively appears
            # to improve subsequent disparity calculation
            grayL = np.power(grayL, 0.75).astype('uint8')
            grayR = np.power(grayR, 0.75).astype('uint8')

            # compute disparity image from undistorted and rectified versions
            disparity_UMat = stereoProcessor.compute(cv2.UMat(grayL),cv2.UMat(grayR))
            disparity = cv2.UMat.get(disparity_UMat)

            speckleSize = math.floor((width * height) * 0.0005)
            maxSpeckleDiff = (8 * 16) # 128

            cv2.filterSpeckles(disparity, 0, speckleSize, maxSpeckleDiff)

            # scale the disparity to 8-bit for viewing
            # divide by 16 and convert to 8-bit image (then range of values should
            # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
            # so we fix this also using a initial threshold between 0 and max_disparity
            # as disparity=-1 means no disparity available
            _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
            disparity_scaled = (disparity / 16.).astype(np.uint8)

            disparity_to_display = (disparity_scaled * (256. / max_disparity)).astype(np.uint8)

            # Run tracking
            results = model.track(frameL, conf=0.5, show=False, persist=True, classes=[0])

            frame = counter.start_counting(frame, results)

            # View results
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    # class name
                    cls = int(box.cls[0])
                    className = classNames[cls]
                    if className == "person":
                        x, y, _, _ = box.xywh[0]
                        x, y = int(x), int(y)

                        # Calculate distance to object
                        distance = calc_dist(x, y, disparity_scaled, fx, B)

                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # put box in cam
                        cv2.rectangle(frameL, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(frameL, className + ': {0:.3f}'.format(distance / 1000), org, font, fontScale, color, thickness)
           
            # display disparity 
            cv2.imshow(windowNameD, disparity_to_display)
            # display input image (combined left and right)
            cv2.imshow(windowName, frame)
            if cv2.waitKey(1) == ord('q'):
                keep_processing = False       
            
        # release camera
        zed_cam.release()
        cv2.destroyAllWindows()

main()