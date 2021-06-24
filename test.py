# ---------------------------------------------
# Project: Engagement detection using emotional information
#  written by Nguyen Tan Ho
# 2021/04/15
# ---------------------------------------------

# -Import common packages-
import os
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
os.environ['DISPLAY'] = ':1'
# ------------
import Api_tools.IO_file as IO_file     # import api_tools
from tools import *                     # import tools
from initialization import *            # import initialization
# -Import Objects-
from Objects import config_
from Objects import inputSource_
from Objects import faceDetection_
from Objects import emotionDetection_
from Objects import face_landMarkDetection_
from Objects import eyeGazeEstimation_
# -Import Algorithms-
import testModule

# ------------




def main():
    config = initialize_Config()
    inputSource = initialize_inputSource(config)

    # Intialize techniques
    # --Face detector
    faceDetector_Config, faceDetector = initialize_faceDetector(config)
    # --Emotion detector
    emotionDetector_Config, emotionDetector = initialize_emotionDetector(config)
    # --Facial Landmark detector
    face_landMarkDetection_Config, face_LandMarkDetector = initialize_face_landMarkDetector(config)

    # Initialize queue and threading
    faces_per_frame_queue = queue.Queue()



    # ----Setting-----
    currentTime = 0
    previousTime = 0
    first_frame = True
    if config.write2File is True:
        _fourcc = cv2.VideoWriter_fourcc("F","M","P","4")
        out = cv2.VideoWriter(config.outputFile_path, _fourcc, float(inputSource.fps), (int(inputSource.width),int(inputSource.height)))
    
    video = cv2.VideoCapture(inputSource.path) # read input Source
    # Run the loop
    count = 0
    while True:
        ret, img = video.read()
        if ret == True:
            img_show = img.copy()
            count +=1
            # Face detection 
            faces = faceDetector_Config.detect_faces(faceDetector, img)         
            print("Found {0} faces!".format(len(faces)))


            # check if first_frame
            if first_frame is True:
                first_frame = False
                # if face exists in dataset
                # Update....
                # if face doesn't exists in dataset
                #initialize_random_student(faces) # temp
                pass

            # Emotion detection
            for index_face, face in enumerate(faces):
                [x,y,w,h] = face["box"] if faceDetector_Config.type=="mtcnn" else list(face)
                ROI_crop =  img[y:y+h, x:x+w]
                ROI_gray = cv2.cvtColor(ROI_crop, cv2.COLOR_BGR2GRAY)
                # Emotion detection 
                emotion = emotionDetector_Config.detect_emotions(emotionDetector, ROI_gray)
                # Draw on frame
                if config.drawOnFrame is True:
                    draw_BoundingBox(img_show, x, y, w, h, (0,255,255))
                    draw_EmotionLabel(img_show,x, y, w, h, emotion, emotionDetector_Config.emotion_color_dict[emotion])
                    #draw_EyeGageLine()
            

            # Landmasks detection
            faces_data = face_landMarkDetection_Config.detect_landMarks(face_LandMarkDetector,faces,img)
            # EyeGaze estimation
            eye_centers=[]
            for index,POI in enumerate(faces_data):
                eye_corners=POI[2]
                eye_center=eyeGazeEstimation_.getEyePos(eye_corners,img)
                eye_centers.append(eye_center) 
                #print(len(eye_centers))
            
            for index,POI in enumerate(faces_data):
                viewPoint=eyeGazeEstimation_.getCoordFromFace(POI[0],eye_centers[index])
                cv2.line(img_show,(int(eye_centers[index][0][0]),int(eye_centers[index][0][1])),(int(eye_centers[index][0][0]-viewPoint[0]),int(eye_centers[index][0][1]-viewPoint[1])),(0,255,0),4, -1) # detect landmasks



            # calculate the running time
            currentTime = time.time()
            delay2Frame = float(currentTime-previousTime)
            previousTime = currentTime
            print("Delay: {}".format(round(delay2Frame,2)))
            fps = int(1/delay2Frame)
            fps_str = str(fps)+"|"+str(int(inputSource.fps))

            
            if config.drawRunningTime is True:
                draw_runningTime(img_show, str(fps_str), inputSource.width, inputSource.height)
            # Frame Post-Process
            if config.write2File is True:
                out.write(img_show)
            if count >=360:
                break
            if config.showFrame is True:
                cv2.imshow('frame',img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    


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