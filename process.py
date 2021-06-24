import os
import math
import cv2
import time
import numpy as np
import os
import os.path as osp
import sys
import tensorflow as tf
import streamlit as st
import pandas as pd
import time
import pafy
import dlib
import queue
import threading
from multiprocessing import Process, Lock, Queue
import multiprocessing
from math import atan2,degrees

os.environ['DISPLAY'] = ':1'
# ------------
import Api_tools.IO_file as IO_file     # import api_tools
from tools import *                     # import tools
from initialization import *            # import initialization
from Objects import *                   # import objects
from process import *                   # import processes
# ------------

def emotion_detection_module(config, faces_queue, emotions_queue):
    print("intialize emotionDetector")
    emotionDetector_Config, emotionDetector = initialize_emotionDetector(config)
    while True:
        face_dict = faces_queue.get()
        emotion = emotionDetector_Config.detect_emotions(emotionDetector, ROI_gray)
        print(emotion)
        emotion_dict = face_dict.update({"emotion":emotion})
        emotions_queue.put(emotion_dict)

def eye_estimation_module(config, faces_queue, eye_gaze_queue):
    print("intialize eyeGaze estimation")
    face_landMarkDetection_Config, face_LandMarkDetector = initialize_face_landMarkDetector(config)
    while True:
        face_dict = faces_queue.get()
        face_data = face_landMarkDetection_Config.detect_landMarks(face_LandMarkDetector, face_dict["face"], face_dict["frame"])
        eye_corners = face_data[2]
        eye_center = eyeGazeEstimation_.getEyePos(eye_corners,face_dict["frame"])
        viewPoint = eyeGazeEstimation_.getCoordFromFace(face_data[0],eye_center)
        eye_centers_ord = (int(eye_center[0][0]),int(eye_center[0][1]))
        eye_view_ord = (int(eye_center[0][0]-viewPoint[0]),int(eye_center[0][1]-viewPoint[1]))
        degree = eyeGazeEstimation_.AngleBtw2Points(eye_centers_ord, eye_view_ord)
        distance = int(math.sqrt( ((eye_centers_ord[0]-eye_view_ord[0])**2)+((eye_centers_ord[1]-eye_view_ord[1])**2)))
        print(distance)

def drawing_frame_module(config, display_queue, emotionDetector_Config):
    print("intialize drawing frame module")
    while True:
        display_frame = display_queue.get()
        # DRAW ON FRAME
        if config.turnOn_emotion_detection:
            draw_EmotionLabel(img_show,x, y, w, h, emotion, emotionDetector_Config.emotion_color_dict[emotion])
        if config.turnOn_eyeGaze_estimation:
            # Draw EyeGaze line
            if attention == "focus":
                draw_AttentionLabel(img_show, x, y, w, h, "looking at screen", (0, 255, 0))
            else:
                draw_AttentionLabel(img_show, x, y, w, h, "looking away from the screen", (0, 0, 255))
            draw_EyeGageLine(img_show, eye_centers_ord, eye_view_ord, (0, 255, 255))
        if config.turnOn_face_identification:
            # Draw student_name label
            draw_NameLabel(img_show, x, y, w, h, "21R"+"%03d" %int(student._name), (255,255,255))
                    
        if attention == "focus" :
            #draw_BoundingBox(img_show, x, y, w, h, (0,255,255))
            draw_BoundingBox(img_show, x, y, w, h, (0,255,0))
        else:
            draw_BoundingBox(img_show, x, y, w, h, (0,0,255))
