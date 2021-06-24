# ---------------------------------------------
# Project: Engagement detection using emotional information
#  written by Nguyen Tan Ho
# 2021/04/15
# ---------------------------------------------

# -Import common packages-
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
from math import atan2,degrees

os.environ['DISPLAY'] = ':1'
# ------------
import Api_tools.IO_file as IO_file     # import api_tools
from tools import *                     # import tools
from initialization import *            # import initialization
# -Import Objects-
from Objects import config_
from Objects import inputSource_
from Objects import control_
from Objects import faceDetection_
from Objects import emotionDetection_
from Objects import face_landMarkDetection_
from Objects import eyeGazeEstimation_
from Objects import engagementDetection_
from Objects import student_
from Objects import faceIdentification_
# -Import Algorithms-
import testModule

# ------------




def main():
    config = initialize_Config()
    inputSource = initialize_inputSource(config)
    control = initialize_Control(config)
    # Intialize techniques
    # --Face detector
    faceDetector_Config, faceDetector = initialize_faceDetector(config)
    # --Emotion detector
    emotionDetector_Config, emotionDetector = initialize_emotionDetector(config)
    # --Facial Landmark detector
    face_landMarkDetection_Config, face_LandMarkDetector = initialize_face_landMarkDetector(config)
    # Initialize queue and threading
    faces_per_frame_queue = queue.Queue()

    #initialize temp
    """    
    status_eyeGaze_count = 0
    max_distance = 0
    total_distance = 0
    """
    # initialize student_list
    student_list =[]

    # ----Setting-----
    currentTime = 0
    previousTime = 0
    if config.write2File is True:
        _fourcc = cv2.VideoWriter_fourcc("F","M","P","4")
        out = cv2.VideoWriter(config.outputFile_path, _fourcc, float(inputSource.fps), (int(inputSource.width),int(inputSource.height)))
    
    video = cv2.VideoCapture(inputSource.path) # read input Source
    # Run the loop
    frame_process_jump = int(int(inputSource.fps) * float(config.time_process))
    while True:
        ret, img = video.read()
        if ret == True:
 
            img_show = img.copy()
            if control._frame_idx % frame_process_jump == 0 :
                    
                # Face detection 
                faces = faceDetector_Config.detect_faces(faceDetector, img)         
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                #faces_per_frame_queue.put_nowait(faces)                     
                for index_face, face in enumerate(faces):
                    control._count_face+=1
                    [x,y,w,h] = face["box"] if faceDetector_Config.type=="mtcnn" else list(face)
                    ROI_crop =  img[y:y+h, x:x+w]
                    ROI_gray = cv2.cvtColor(ROI_crop, cv2.COLOR_BGR2GRAY)
                    emotion = None
                    attention = None
                    student = None
                    if config.turnOn_face_identification:
                        if student_list ==[]:
                            # initialize student
                            student = student_.Student(str(control._count_face), str(control._count_face))
                            face_coord = [x,y,w,h]
                            student._face_point = ((face_coord[0]+face_coord[2])//2,(face_coord[1]+face_coord[3])//2)
                            student._face_region = ROI_gray
                            student_list.append(student)
                        else:
                            #Face identification
                            student = faceIdentification_.identify_face(control, ROI_gray, [x,y,w,h], student_list)
                    
                    # ANALYSE
                    if config.turnOn_emotion_detection:
                        #Emotion detection
                        emotion = emotionDetector_Config.detect_emotions(emotionDetector, ROI_gray)

                    if config.turnOn_eyeGaze_estimation:
                        # Landmasks detection
                        face_data = face_landMarkDetection_Config.detect_landMarks(face_LandMarkDetector, face, gray_img)
                        # EyeGaze estimation
                        eye_corners=face_data[2]
                        eye_center=eyeGazeEstimation_.getEyePos(eye_corners,img)
                        viewPoint=eyeGazeEstimation_.getCoordFromFace(face_data[0],eye_center)
                        eye_centers_ord = (int(eye_center[0][0]),int(eye_center[0][1]))
                        eye_view_ord = (int(eye_center[0][0]-viewPoint[0]),int(eye_center[0][1]-viewPoint[1]))
                        degree = eyeGazeEstimation_.AngleBtw2Points(eye_centers_ord, eye_view_ord)
                        #cv2.putText(img_show, str(int(degree)), eye_centers_ord, cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,255) ,1, cv2.LINE_4)
                        distance = int(math.sqrt( ((eye_centers_ord[0]-eye_view_ord[0])**2)+((eye_centers_ord[1]-eye_view_ord[1])**2)))
                        student._total_distance +=  distance
                        
                        if (int(degree) >= 35 and int(degree) <=155) or distance < (student._total_distance/(control._frame_idx+1)):
                            attention = "focus"
                            student._status_eyeGaze_count = 0
                            print("focus - {}".format(attention))
                        else:
                            if student._status_eyeGaze_count <=3:
                                attention = "focus"
                                student._status_eyeGaze_count +=1
                            else:
                                attention = "distracted"
                            print("distracted - {}".format(attention))
                        
                    # CALCULATE ENGAGEMENT SCORE
                    #engagement_level = engagementDetection_.detect_engagement(emotion, attention)
                    #print(engagement_level)

                    # DRAW ON FRAME
                    if config.drawOnFrame is True:
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
                    
                # calculate the running time
                currentTime = time.time()
                delay2Frame = float(currentTime-previousTime)
                previousTime = currentTime
                #print("{}: {}".format(control._frame_idx, round(delay2Frame,2)))
                fps = int(1/delay2Frame)
                fps_str = str(fps)+"|"+str(int(inputSource.fps))

                if config.drawRunningTime is True:
                    draw_runningTime(img_show, str(fps_str), inputSource.width, inputSource.height)
            # Frame Post-Process
            if config.write2File is True:
                out.write(img_show)
            if control._frame_idx >=50000 or control._frame_idx > (int(inputSource.length)-int(inputSource.fps)):
                break
            if config.showFrame is True:
                cv2.imshow('frame',img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            control._frame_idx +=1

    # Release all 
    video.release()
    if config.write2File is True:
        out.release()
    if config.showFrame is True:
        cv2.destroyAllWindows()



if __name__=='__main__':
    main()

    # -------Test----------
    # Source
    #source_video = "/home/tigerzing/Documents/input.mp4"
    #source_video = "https://www.youtube.com/watch?v=Nu4DXycNzpQ"
    source_video = 0
    
    # Techniques
    techniques = {
        "Face_Detection": "haarcascade_faceDetection", # mtcnn | haarcascade_faceDetection
        "LandMark_Detection": "68_landmarks", # 68_landmarks
        "Emotion_Detection": "haarcascade_emotionDetection", # haarcascade_emotionDetection
        "EyeGaze_Detection": "pnp_algorithm" #pnp_algorithm
    } 

    # Testing section
    #testModule.FaceDetection(source_video, techniques) 
    #testModule.EmotionDetection(source_video, techniques)
    #testModule.LandMaskDetection(source_video, techniques)
    #testModule.EyeGazeEstimation(source_video, techniques)