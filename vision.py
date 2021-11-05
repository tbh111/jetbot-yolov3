import sys
import cv2
import cv2.aruco as aruco
import numpy as np
#import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.yolov3 import TrtYOLOv3
from utils.camera import Camera
from utils.visualization import open_window, show_fps, record_time, show_runtime
from utils.engine import BBoxVisualization
from config.config import writeCameraParam, readCameraParam


WINDOW_NAME = 'TensorRT YOLOv3 Detector'
INPUT_HW = (300, 300)
SUPPORTED_MODELS = [
    'ssd_mobilenet_v2_coco'
]


class State():
    init = 0
    search = 1
    move = 2
    process = 3
    finish = 4
    stop = 5
    debug = 100

class Labels():
    apple = 7
    banana = 2
    orange = 3
    # test = 7

class Vision:
    def __init__(self, args):
        self.cam = Camera(args)
        self.yolo = Yolo_detector(args)
        self.aruco = Aruco_detector()
        self.cam.open()
        if not self.cam.is_opened:
            sys.exit('[INFO]  Failed to open camera!')
        self.cam.start()
        print('[INFO]  Camera: starting')
        open_window(WINDOW_NAME, args.image_width, args.image_height,
                    'TensorRT YOLOv3 Detector')

    def detect(self, state, curr_target):
        src = self.cam.read()
        box = []
        flag = False
        if src is not None:
            # img = img.copy()
            if state == State.search or state == State.move:
                img, loc, flag = self.aruco.detect(src, curr_target)
                return img, loc, flag
            elif state == State.process:
                # img_1 = cv2.resize(src, (300, 300))
                img, box, flag = self.yolo.detect(src, curr_target, conf_th=0.3)
                return img, box, flag
            elif state == State.debug:
                self.yolo.debug = True
                img = self.yolo.detect(src, curr_target, conf_th=0.3)
                return img

            else:
                pass
        else:
            return src, box, flag 


    def unInit(self):
        self.cam.stop()
        self.cam.release()
        print('[INFO] Camera released')


class Yolo_detector:
    def __init__(self, args):
        cls_dict = get_cls_dict('coco')
        yolo_dim = int(args.model.split('-')[-1])  # 416 or 608
        self.trt_yolov3 = TrtYOLOv3(args.model, (yolo_dim, yolo_dim))
        self.vis = BBoxVisualization(cls_dict)
        self.runtime = args.runtime
        self.debug = False

    def detect(self, img, curr_target, conf_th=0.3):
        timer = cv2.getTickCount()
        if img is not None:
            if self.runtime:
                boxes, confs, label, _preprocess_time, _postprocess_time,_network_time = self.trt_yolov3.detect(img, conf_th)
                img, _visualize_time = self.vis.draw_bboxes(img, boxes, confs, label)
                time_stamp = record_time(_preprocess_time, _postprocess_time, _network_time, _visualize_time)
                show_runtime(time_stamp)
            else:
                boxes, confs, label, _, _, _ = self.trt_yolov3.detect(img, conf_th)
                img, _ = self.vis.draw_bboxes(img, boxes, confs, label)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            img = show_fps(img, fps)
            if self.debug:
                return img
            # print('label',label, len(label))
            box = []
            flag = False
            for i in range(0,len(label)):
                if label[i] == 47:# curr_target:
                    flag = True
                    box = boxes[i]
                    break
                else:
                #     flag = False
                #     box = []
                    pass
            # if label.any() == [47]:
            #     flag = True
            
            return img, box, flag
        else:
            return img, [], False

class Aruco_detector:
    def __init__(self):
        self.dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
        self.markerLength = 13
        self.camMat, self.distCoeff = readCameraParam('./config/cameraConfig.yaml')
    
    def estimatePose(self,corners):
        rvecs = np.zeros(3)
        tvecs = np.zeros(3)
        loc = np.zeros(3)
        # print('r:',rvecs)
        # print('t:',tvecs)
        # print('corners:',corners)
        # print('length:',self.markerLength)
        # print('camMat:',self.camMat)
        # print('dist:',self.distCoeff)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.markerLength, self.camMat, self.distCoeff)
        # print('r:',rvecs)
        # print('t:',tvecs)
        rMat, _ = cv2.Rodrigues(rvecs)
        # tMat, _ = cv2.Rodrigues(tvecs)
        # print('rMat:',rMat)
        # print('tMat:',tMat)
        loc = np.dot(-np.linalg.inv(rMat), tvecs[0].T)
        print('dis:',loc)
        return loc

    
    def detect(self, src, target):
        if src is None:
            return src, np.zeros(3), False
        elif target == -1:
            return src, np.zeros(3), False
        else:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.dict)
            flag = False
            box = []
            loc = np.zeros(3)
            if ids is not None:
                #aruco.drawDetectedMarkers(src, corners, ids)
                for i in range(0, len(ids)):
                    if ids[i] == target:
                        flag = True
                        box = corners[i]
                        loc = self.estimatePose(box)
                        aruco.drawDetectedMarkers(src, corners, ids)
                        break
                    else:
                        aruco.drawDetectedMarkers(src, corners, ids)


                return src, loc, flag
            else:
                return src, loc, flag





            

