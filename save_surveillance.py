# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import numpy as np


from darkflow.defaults import argHandler  # Import the default arguments
from darkflow.net.build import TFNet
import os

from datetime import datetime

import csv

from sort import Sort
Tracker = Sort()

csv_file = open("Video_{}.csv".format(datetime.now().strftime("%d_%m_%Y_%H_%M_%S")), 'w')
csv_writer = csv.writer(csv_file)

track_color = {'small-car': (0, 0, 0), 'big-car': (255, 0, 0), 'bus': (0, 255, 0), 'truck': (0, 0, 255),
 'three-wheeler': (255, 255, 0), 'two-wheeler': (0, 255, 255), 'lcv': (255, 0, 255), 'bicycle': (255, 255, 255), 'people': (255, 127, 255), 'auto-rickshaw': (127, 127, 127)}

trackObj = ['small-car',
                      'big-car',
                      'bus',
                      'truck',
                      'three-wheeler',
                      'two-wheeler',
                      'lcv',
                      'bicycle',
                      'people',
                      'auto-rickshaw']

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized


def manual_setting():
    FLAGS = argHandler()
    FLAGS.setDefaults()

    FLAGS.demo = "Video/20180725_1320.mp4"  # Initial video file to use, or if camera just put "camera"
    FLAGS.model = "cfg/yolo_smartcity.cfg"  # tensorflow model
    FLAGS.load = 24304  # 37250  # tensorflow weights
    # FLAGS.pbLoad = "tiny-yolo-voc-traffic.pb" # tensorflow model
    # FLAGS.metaLoad = "tiny-yolo-voc-traffic.meta" # tensorflow weights
    FLAGS.threshold = 0.3  # threshold of decesion confidance (detection if confidance > threshold )
    FLAGS.max_gpu_usage = 0
    FLAGS.number_of_parallel_threads = int(os.environ.get("NO_OF_THREADS", 2))
    FLAGS.gpu = FLAGS.max_gpu_usage / FLAGS.number_of_parallel_threads  # how much of the GPU to use (between 0 and 1) 0 means use cpu
    FLAGS.track = True  # wheither to activate tracking or not
    FLAGS.trackObj = ['small-car',
                      'big-car',
                      'bus',
                      'truck',
                      'three-wheeler',
                      'two-wheeler',
                      'lcv',
                      'bicycle',
                      'people',
                      'auto-rickshaw']
    # ['car', 'bus',
    #               'motorbike']  # ['Bicyclist','Pedestrian','Skateboarder','Cart','Car','Bus']  the object to be tracked
    # FLAGS.trackObj = ["person"]
    FLAGS.saveVideo = False  # whether to save the video or not
    FLAGS.BK_MOG = False  # activate background substraction using cv2 MOG substraction,
    # to help in worst case scenarion when YOLO cannor predict(able to detect mouvement, it's not ideal but well)
    # helps only when number of detection < 3, as it is still better than no detection.
    FLAGS.tracker = "sort"  # which algorithm to use for tracking deep_sort/sort (NOTE : deep_sort only trained for people detection )
    FLAGS.skip = 0  # how many frames to skipp between each detection to speed up the network
    FLAGS.csv = True  # whether to write csv file or not(only when tracking is set to True)
    FLAGS.display = True  # display the tracking or not
    FLAGS.testing = True
    return FLAGS


def Initialize(FLAGS):
    tfnet = TFNet(FLAGS)
    return tfnet


#def Play_video(tfnet):
 #   tfnet.camera()
    # exit('Demo stopped, exit.')

Flags = manual_setting()

net = Initialize(Flags)

FILE_OUTPUT = 'output_TT.mp4'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture('rtsp://live:Smart#123@192.168.0.100/rtsp_tunnel?h26x=4&line=1&inst=2')



currentFrame = 0

# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

line_coordinate = [3, 649, 1284, 190]
detections = []
b_results = []

Tracker.frame_width = width
Tracker.frame_height = height
Tracker.frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
Tracker.line_coordinate = line_coordinate


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(width),int(height)))

# while(True):
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    h, w, _ = frame.shape
    thick = int((h + w) // 300)

    if ret == True:
        # Handles the mirroring of the current frame
        #frame = cv2.flip(frame,1)
        result = net.return_predict(frame)
        #print(len(result))
        print(result)
        for x in result:
            detections.append([x['topleft']['x'],x['topleft']['y'],x['bottomright']['x'],x['bottomright']['y'],x['confidence']])
            b_results.append([x['topleft']['x'],x['topleft']['y'],x['bottomright']['x'],x['bottomright']['y'],x['label']])
        #print(detections)
        counter, detections, boxes_final, imgcv = result
        #print(counter)
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.line(frame, (line_coordinate[0], line_coordinate[1]), (line_coordinate[2], line_coordinate[3]),
                 (0, 0, 255), 6)
        trackers = Tracker.update(np.asarray(detections), b_results,  csv_file, csv_writer, timestamp, testing=True)
        #for track in trackers:
        
        #trackers = Tracker.update(detections, boxes_final, csv_file, csv_writer, timestamp, self.options.testing)
        for track in trackers:
                print(track)
                try:
                    bbox = [int(track[0]), int(track[1]), int(track[2]), int(track[3])]
                    for box in b_results:
                        if (((box[2]) - 30) <= int(bbox[1]) <= ((box[2]) + 30) and ((box[0]) - 30) <= bbox[0] <= (
                                (box[0]) + 30)):
                            global v_name
                            v_name = box[4]
                            #print(v_name)

                    #if v_name == 'car':
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  track_color[box[4]], thick // 3)
                except Exception as e:
                    print(str(e))
        vehicle_count = Tracker.vehicle_count
        #print(vehicle_count)
        BASE_Y = 30
        Y_WIDTH = 50
        FONT = cv2.FONT_HERSHEY_COMPLEX
        FONT_SCALE = 0.6
        FONT_COLOR = (0, 0, 0)
        RATIO_OF_DIALOG_BOX = 0.5
        #frame = self.current_frame
        # frame = cv2.resize(self.current_frame, None, fx=self.VIDEO_SCALE_RATIO, fy=self.VIDEO_SCALE_RATIO,
        #                    interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        width = frame.shape[1]
        height = frame.shape[0]
        b_width = round(frame.shape[1] * RATIO_OF_DIALOG_BOX)
        b_height = height
        blank_image = np.zeros((height, b_width, 3), np.uint8)
        blank_image[np.where((blank_image == [0, 0, 0]).all(axis=2))] = [240, 240, 240]
        overlay = np.zeros((height, width, 4), dtype='uint8')
        img_path = 'icons/bosch.png'
        logo = cv2.imread(img_path, -1)
        watermark = image_resize(logo, height=50)
        watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
        watermark_h, watermark_w, watermark_c = watermark.shape
        for i in range(0, watermark_h):
            for j in range(0, watermark_w):
                if watermark[i, j][3] != 0:
                    offset = 10
                    h_offset = height - watermark_h - offset
                    w_offset = height - watermark_w - offset
                    overlay[10 + i, 10 + j] = watermark[i, j]
        cv2.putText(frame, "SmartCity Solutions", (width - int(width * 0.25), round(height * 0.1)), FONT, 1,
                        (255, 255, 255), 2)
        cv2.addWeighted(overlay, 1, frame, 1, 0, frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        GRID_WIDTH = 200
        PADDING = 10
        INITIAL = PADDING + GRID_WIDTH + 50
        cv2.rectangle(blank_image, (PADDING, PADDING), (b_width - PADDING, b_height - PADDING), (0, 0, 0), 1)
        cv2.line(blank_image, (INITIAL, 10), (INITIAL, b_height - PADDING), (0, 0, 0), 1)
        cv2.line(blank_image, (INITIAL + GRID_WIDTH * 1, 10), (INITIAL + GRID_WIDTH * 1, b_height - PADDING),
                         (0, 0, 0), 1)
        cv2.line(blank_image, (INITIAL + GRID_WIDTH * 2, 10), (INITIAL + GRID_WIDTH * 2, b_height - PADDING),
                         (0, 0, 0), 1)
        # # Initial Data
        cv2.putText(blank_image, 'Vehicle Type', (50, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)
        cv2.putText(blank_image, 'In Flow', (330, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)
        cv2.putText(blank_image, 'Out Flow', (530, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)
        cv2.putText(blank_image, 'Total', (750, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)

        for id, obj in enumerate(trackObj):
                    cv2.putText(blank_image, obj, (50, BASE_Y + Y_WIDTH * (1 + id)), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, str(vehicle_count[obj][1]), (330, BASE_Y + Y_WIDTH * (1 + id)), FONT,
                                FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, str(vehicle_count[obj][0]), (530, BASE_Y + Y_WIDTH * (1 + id)), FONT,
                                FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, str(vehicle_count[obj][0] + vehicle_count[obj][1]),
                                (750, BASE_Y + Y_WIDTH * (1+ id)), FONT, FONT_SCALE, FONT_COLOR, 1 )
        img = np.column_stack((frame, blank_image))
        # Saves for video
        out.write(img)

        # Display the resulting frame
        cv2.imshow('frame',img)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
csv_file.close()
cap.release()
out.release()
cv2.destroyAllWindows()

# Potential Error:
# OpenCV: Cannot Use FaceTime HD Kamera
# OpenCV: camera failed to properly initialize!
# Segmentation fault: 11
#
# Solution:
# I solved this by restarting my computer.
# http://stackoverflow.com/questions/40719136/opencv-cannot-use-facetime/42678644#42678644
