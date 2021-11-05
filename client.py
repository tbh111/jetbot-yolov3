from jetbot import Robot
from utils.PID import IncrementalPID, PositionalPID
from utils.servoserial import ServoSerial
from utils.camera import add_camera_args
import time
import math

from vision import Vision, State

import argparse
import numpy as np
import cv2

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLOv3 model on Jetson Family')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, default='yolov3-416',
                        choices=['yolov3-288', 'yolov3-416', 'yolov3-608',
                                'yolov3-tiny-288', 'yolov3-tiny-416'])
    parser.add_argument('--runtime', action='store_true',
                        help='display detailed runtime')
    args = parser.parse_args()
    return args


class Client(object):
    def __init__(self, target_array):
        self.state = State.init
        self.robot = Robot()
        self.img = np.zeros((480, 640))

        self.xservo_pid = PositionalPID(1.9, 0.3, 0.35)
        self.yservo_pid = PositionalPID(1.5, 0.2, 0.3)
        self.servo = ServoSerial()
        self.xpos = 2100
        self.ypos = 2048
        self.servo.Servo_serial_double_control(1, self.xpos, 2, self.ypos)
        self.panpilt_dir = True # clockwise search
        self.search_delta = 100
        self.search_count = 0

        self.move_distance = np.zeros(3)

        self.args = parse_args()
        self.vision = Vision(self.args)
        self.state = State.search
        self.target = target_array
        if len(self.target) > 0:
            self.curr_target = self.target[0]
        else:
            self.curr_target = -1

    def switchState(self):
        if self.state == State.search: # aruco function
            self.img, _, flag = self.vision.detect(self.state, self.curr_target)
            if flag: # detected target aruco, move to it
                print('done')
                self.state = State.move
            else: # no aruco or wrong aruco, pantilt will revolve periodly
                # self.pantiltSearch()
                self.state = State.search
        
        elif self.state == State.move:
            self.img, dis, flag = self.vision.detect(self.state, self.curr_target)
            # if flag: # move to target
            # if flag:
            self.moveToTarget(dis, 1)
            self.move_distance = -1*dis
            self.state = State.process
            print('move done, enter yolo process')
            # else: # target lost, return to search state
                # self.state = State.search
            # if box is not None:
            #     print('[INFO] Find target')
            #     self.state = State.search
                # self.moveToTarget(distance, center, 1.0)

        elif self.state == State.debug:
            self.img = self.vision.detect(self.state, self.curr_target)

        elif self.state == State.process: # yolo function
            self.ypos = 1500
            self.servo.Servo_serial_control(2, self.ypos)

            while True:
                self.img, center, flag = self.vision.detect(self.state, self.curr_target)
                print(center)
                if flag: # find target, save image, then return back to the horizonal position
                    x = 320 - (center[2] + center[0])/2
                    y = 240 - (center[3] + center[1])/2
                    #print(x,y)
                    cv2.circle(self.img, 
                        (int((center[2] + center[0])/2),
                        int((center[3] + center[1])/2)), 10, (255,0,0), 2)
                    self.pantiltControl(x, y)
                    if math.sqrt(x**2+y**2) < 100:
                        self.search_count = 0
                        self.state = State.debug
                        
                        break
                else:
                    self.pantiltSearch()
                    self.search_count = self.search_count + 1
                    if self.search_count > 3*(4200 / self.search_delta):
                        print('no target, return back')
                        # self.state = State.finish
                        self.state = State.process
                        break
                    else:
                        print('searching yolo object')
                        self.state = State.process                    


            # if flag: # find target, save image, then return back to the horizonal position
            #     x = 320 - (center[2] + center[0])/2
            #     y = 240 - (center[3] + center[1])/2
            #     #print(x,y)
            #     cv2.circle(self.img, 
            #         (int((center[2] + center[0])/2),
            #         int((center[3] + center[1])/2)), 10, (255,0,0), 2)
            #     self.pantiltControl(x, y)
            #     while True:
            #         self.img, center, flag = self.vision.detect(self.state, self.curr_target)
            #         if flag:
            #             x = 320 - (center[2] + center[0])/2
            #             y = 240 - (center[3] + center[1])/2
            #             self.pantiltControl(x, y)
            #         if math.sqrt(x^2+y^2) < 100:
            #             break

            #     self.state = State.finish

            #     # self.target.remove()
            #     self.state = State.search
            #     self.search_count = 0

            # else:
            #     self.pantiltSearch()
            #     self.search_count = self.search_count + 1
            #     if self.search_count > 3*(4200 / self.search_delta):
            #         print('no target, return back')
            #         # self.state = State.finish
            #         self.state = State.process
            #     else:
            #         print('searching yolo object')
            #         self.state = State.process

        elif self.state == State.finish:
            self.xpos = 2100 # reset the panpilt angle
            self.ypos = 2048
            self.servo.Servo_serial_double_control(1, self.xpos, 2, self.ypos)
            self.moveToTarget(self.move_distance, 1)


        elif self.state == State.stop:
            self.stopClient()
            self.vision.unInit()

    def stopClient(self):
        self.state = State.stop

    def moveToTarget(self, distance, speed):
        time_x = distance[0] / speed
        # time_y = distance[1] / speed
        time_z = (distance[2] - 10) / speed
        '''
        # first, align the bot with the horizional axis(x-axis)
        if time_x > 0 : # turn right
            self.robot.right(speed=speed)
            time.sleep(0.5) # time for bot to rotate 90 degree
            self.robot.forward(speed=speed)
            time.sleep(time_x)
            self.robot.left(speed=speed)
            time.sleep(0.5)
            self.xpos = 2100 # reset the panpilt angle
            self.ypos = 2048
            self.servo.Servo_serial_double_control(1, self.xpos, 2, self.ypos)
            time.sleep(0.5)
            # self.robot.forward(speed=speed)
            # time.sleep(time_z)
            # self.robot.stop()
        elif time_x < 0: # turn left
            self.robot.left(speed=speed)
            time.sleep(0.5)
            self.robot.forward(speed=speed)
            time.sleep(time_x)
            self.robot.right(speed=speed)
            time.sleep(0.5)
            self.xpos = 2100 # reset the panpilt angle
            self.ypos = 2048
            self.servo.Servo_serial_double_control(1, self.xpos, 2, self.ypos)
            time.sleep(0.5)
        '''
        _, dis, _ = self.vision.detect(self.state, self.curr_target)
        while dis[2] > 25:
            self.robot.forward(speed=speed)
            time.sleep(1)
            _, dis, _ = self.vision.detect(self.state, self.curr_target)
        if dis[2] < 25:
            print('reached target')
            self.robot.stop()
            # self.robot.forward(speed=speed)
            # time.sleep(time_z)
            # self.robot.stop()

        
        # if target[0] > 360 and target[0] < 720:
        #     self.robot.right(speed=speed)
            
        # elif target[0] < 360 and target[0] > 0:
        #     self.robot.left(speed=speed)
        
        # if distance > 100 and distance < 1000:
        #     self.robot.forward(speed=speed)
        # elif distance > 0 and distance < 50:
        #     self.robot.backward(speed=speed)
        # else:
        #     print('[INFO] Reached target, start yolo process')
        #     self.state = State.process

    def pantiltControl(self, x, y):
        #Proportion-Integration-Differentiation算法
        # 输入X轴方向参数PID控制输入
        # self.xservo_pid.SystemOutput = x
        # self.xservo_pid.SetStepSignal(150)
        # self.xservo_pid.SetInertiaTime(0.01, 0.006)
        if abs(x) > 80: 
            self.xpos = int(self.xpos + 0.8*x)
        elif abs(x) <= 80 and abs(x) > 10:
            self.xpos = int(self.xpos + 0.1*x)

        # print('x ',target_valuex, int(2100 + self.xservo_pid.SystemOutput))
        # target_valuex = int(self.xpos + self.xservo_pid.SystemOutput)
        # 输入Y轴方向参数PID控制输入
        # self.yservo_pid.SystemOutput = y
        # self.yservo_pid.SetStepSignal(150)
        # self.yservo_pid.SetInertiaTime(0.01, 0.006)
        if abs(y) > 80:
            self.ypos = int(self.ypos + 0.8*y)
        # print('y ',target_valuey, int(2048+self.yservo_pid.SystemOutput))

        # target_valuey = int(self.ypos+self.yservo_pid.SystemOutput)
        # 将云台转动至PID调校位置
        print('x:y',self.xpos, self.ypos)
        self.servo.Servo_serial_double_control(1, self.xpos, 2, self.ypos)

    def pantiltSearch(self):
        if self.xpos >= 3600 or self.xpos <= 600:
            self.panpilt_dir = bool(1-self.panpilt_dir) # invert
        print('dir: ', self.panpilt_dir, 'xpos: ', self.xpos)
        if self.panpilt_dir:
            self.servo.Servo_serial_control(1, self.xpos-self.search_delta)
            self.xpos = self.xpos-self.search_delta
        else:
            self.servo.Servo_serial_control(1, self.xpos+self.search_delta)
            self.xpos = self.xpos+self.search_delta
        time.sleep(0.1)

        
