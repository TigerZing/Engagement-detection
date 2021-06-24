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
import pafy
import dlib
import math
from multiprocessing import Process, Lock
# ------------
import Api_tools.IO_file as IO_file # import api_tools
from tools import *                     # import tools
from initialization import *   
# -Import Objects-
from Objects import config_
from Objects import inputSource_
from Objects import faceDetection_
from Objects import emotionDetection_
from Objects import face_landMarkDetection_
from Objects import eyeGazeEstimation_
# -Import Algorithms-

# ----------------Test ------------------

def FaceDetection(source_video, techniques):
    # --------
    config = initialize_Config()
    config.source_video = source_video
    config.type_faceDetection = techniques["Face_Detection"]
    inputSource = initialize_inputSource(config)
    # Intialize techniques
    # --Face detector
    faceDetector_Config = faceDetection_.FaceDetection()
    faceDetector_Config.process(config)
    faceDetector = faceDetector_Config.intialize()
    # time show calculation
    currentTime = 0
    previousTime = 0
    video = cv2.VideoCapture(inputSource.path) # read input Source
    # Run the loop
    while True:
        ret, img = video.read()
        if ret == True:
            # Face detection
            if faceDetector_Config.type == "haarcascade_faceDetection":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(gray, 1.3, 5)
                faces = list(faces)
            elif faceDetector_Config.type == "mtcnn":
                faces = faceDetector.detect_faces(img)
            for index_face, face in enumerate(faces):
                [x,y,w,h] = face["box"] if faceDetector_Config.type=="mtcnn" else list(face)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            print(len(faces))
            # Calculate the processing time
            currentTime = time.time()
            fps = str(int(1/(currentTime-previousTime)))
            previousTime = currentTime
            fps_str = fps+"|"+str(int(inputSource.fps))
            cv2.putText(img,  fps_str, (int(inputSource.width) - 110, 30), cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,255) ,1, cv2.LINE_4)
            cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

def LandMaskDetection(source_video, techniques):
    config = initialize_Config()
    config.source_video = source_video
    config.type_faceDetection = techniques["Face_Detection"]
    config.type_landMaskDetection = techniques["LandMark_Detection"]
    inputSource = initialize_inputSource(config)
    # Intialize techniques
    # --Face detector
    #detector = dlib.get_frontal_face_detector()
    faceDetector_Config = faceDetection_.FaceDetection()
    faceDetector_Config.process(config)
    faceDetector = faceDetector_Config.intialize()
    # --Landmarks detector
    landMarksDetector_Config = face_landMarkDetection_.Face_LandMarkDetection()
    landMarksDetector_Config.process(config)
    landMarksDetector = landMarksDetector_Config.intialize()
    predictor = dlib.shape_predictor(landMarksDetector_Config.weight_path)

    # time show calculation
    currentTime = 0
    previousTime = 0
    video = cv2.VideoCapture(inputSource.path) # read input Source
    # Run the loop
    while True:
        ret, img = video.read()
        if ret == True:
            if faceDetector_Config.type == "haarcascade_faceDetection":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(gray, 1.3, 5)
                faces = list(faces)
            elif faceDetector_Config.type == "mtcnn":
                faces = faceDetector.detect_faces(img)
            """
            faces = detector(img)
            """
            # detect landmasks
            faces_data = landMarksDetector_Config.detect_landMarks(landMarksDetector,faces,img)
            #print(faces)
            for i in range(len(faces)):
                converted_face = dlib.rectangle(faces[i][0],faces[i][1],faces[i][0]+faces[i][2],faces[i][1]+faces[i][3])
                landmarks = np.matrix([[p.x, p.y] for p in predictor(img, converted_face).parts()])
                for idx, point in enumerate(landmarks):
                    pos = (point[0, 0], point[0, 1])
                    cv2.circle(img, pos, 2, color=(255, 255, 255))
                    cv2.putText(img, str(idx + 1), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

            # calculate the processing time
            currentTime = time.time()
            fps = str(int(1/(currentTime-previousTime)))
            previousTime = currentTime
            fps_str = fps+"|"+str(int(inputSource.fps))
            cv2.putText(img,  fps, (int(inputSource.width) - 110, 30), cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,0) ,2, cv2.LINE_4)
            cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()

def EyeGazeEstimation(source_video, techniques):
    config = initialize_Config()
    config.source_video = source_video
    config.type_faceDetection = techniques["Face_Detection"]
    config.type_landMaskDetection = techniques["LandMark_Detection"]
    config.type_eyeGazeEstimation = techniques["EyeGaze_Detection"]
    inputSource = initialize_inputSource(config)
    # Intialize techniques
    # --Face detector
    #detector = dlib.get_frontal_face_detector()
    faceDetector_Config = faceDetection_.FaceDetection()
    faceDetector_Config.process(config)
    faceDetector = faceDetector_Config.intialize()
    # --Landmarks detector
    landMarksDetector_Config = face_landMarkDetection_.Face_LandMarkDetection()
    landMarksDetector_Config.process(config)
    landMarksDetector = landMarksDetector_Config.intialize()
    # --Eyegaze estimatior
    """
    eyeGazeEstimatior_Config = eyeGazeEstimation_.EyeGazeEstimation()
    eyeGazeEstimatior_Config.process(config)
    eyeGazeEstimatior = eyeGazeEstimatior_Config.intialize()
    """
    # time show calculation
    currentTime = 0
    previousTime = 0
    dist = 0
    video = cv2.VideoCapture(inputSource.path) # read input Source
    # Run the loop
    while True:
        ret, img = video.read()
        if ret == True:
            if faceDetector_Config.type == "haarcascade_faceDetection":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(gray, 1.3, 5)
                faces = list(faces)
            elif faceDetector_Config.type == "mtcnn":
                faces = faceDetector.detect_faces(img)

            # detect landmasks
            faces_data = landMarksDetector_Config.detect_landMarks(landMarksDetector,faces,img)
            eye_centers=[]
            for index,POI in enumerate(faces_data):
                eye_corners=POI[2]
                eye_center=eyeGazeEstimation_.getEyePos(eye_corners,img)
                eye_centers.append(eye_center)

            for index,POI in enumerate(faces_data):
                viewPoint=eyeGazeEstimation_.getCoordFromFace(POI[0],eye_centers[index])
                cv2.line(img,(int(eye_centers[index][0][0]),int(eye_centers[index][0][1])),(int(eye_centers[index][0][0]-viewPoint[0]),int(eye_centers[index][0][1]-viewPoint[1])),(0,255,0),4, -1)
                dist = math.hypot(int(eye_centers[index][0][0]) - int(eye_centers[index][0][0]-viewPoint[0]), int(eye_centers[index][0][1]) - int(eye_centers[index][0][1]-viewPoint[1]))
            # calculate the processing time
            currentTime = time.time()
            fps = float(currentTime-previousTime)
            print("FPS {}".format(round(fps,2)))
            previousTime = currentTime
            cv2.putText(img,  str(dist)+"|"+str(int(inputSource.fps)), (int(inputSource.width) - 110, 30), cv2.FONT_HERSHEY_DUPLEX , 1, (0,255,0) ,2, cv2.LINE_4)
            cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()

def EmotionDetection(source_video, techniques):
    config = initialize_Config()
    config.source_video = source_video
    # modify here
    config.type_faceDetection = techniques["Face_Detection"]
    config.type_landMaskDetection = techniques["LandMark_Detection"]
    config.type_emotionDetection = techniques["Emotion_Detection"]
    # -----------------
    inputSource = initialize_inputSource(config)
    # Intialize techniques
    # --Face detector
    print("intialize faceDetector ...",end=" ")
    faceDetector_Config = faceDetection_.FaceDetection()
    faceDetector_Config.process(config)
    faceDetector = faceDetector_Config.intialize()
    print("OK")

    # --Emotion detector
    print("intialize emotionDetector ...",end=" ")
    emotionDetector_Config = emotionDetection_.EmotionDetection()
    emotionDetector_Config.process(config)
    emotionDetector = emotionDetector_Config.intialize()
    print("OK")


    # time show calculation
    currentTime = 0
    previousTime = 0
    
    video = cv2.VideoCapture(inputSource.path) # read input Source
    # Run the loop
    while True:
        ret, img = video.read()
        if ret == True:
            faces = faceDetector_Config.detect_faces(faceDetector, img)
            for index_face, face in enumerate(faces):
                [x,y,w,h] = face["box"] if faceDetector_Config.type=="mtcnn" else list(face)
                ROI_crop =  img[y:y+h, x:x+w]
                ROI_gray = cv2.cvtColor(ROI_crop, cv2.COLOR_BGR2GRAY)
                emotion = emotionDetector_Config.detect_emotions(emotionDetector, ROI_gray)
                #print(emotion)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
                if y > 50:
                    cv2.rectangle(img,(x, y-45),(x+len(emotion)*20 -len(emotion)*2 ,y-15),(0,255,255),-1)
                    cv2.putText(img,  emotion, (x, y-20), cv2.FONT_HERSHEY_DUPLEX , 1, emotionDetector_Config.emotion_color_dict[emotion] ,2, cv2.LINE_4)
                else: 
                    cv2.rectangle(img,(x, y+h+15),(x+len(emotion)*20 -len(emotion)*2 ,y+h+45),(0,255,255),-1)
                    cv2.putText(img,  emotion, (x, y+h+40), cv2.FONT_HERSHEY_DUPLEX , 1, emotionDetector_Config.emotion_color_dict[emotion] ,2, cv2.LINE_4)
                        # calculate the processing time
            currentTime = time.time()
            delay2Frame = float(currentTime-previousTime)
            previousTime = currentTime
            print("Delay: {}".format(round(delay2Frame,2)))
            fps = int(1/delay2Frame)
            fps_str = str(fps)+"|"+str(int(inputSource.fps))
            cv2.putText(img, fps_str, (int(inputSource.width) - 110, 30), cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,255) ,1, cv2.LINE_4)
            cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()


def Emotion_Calculation(img, faces, faceDetector_Config, emotionDetector_Config, emotionDetector, count_frame):
    print("Emotion_Calculation {}".format(faces[0]))
    for index_face, face in enumerate(faces):
        [x,y,w,h] = face["box"] if faceDetector_Config.type=="mtcnn" else list(face)
        ROI_crop =  img[y:y+h, x:x+w]
        ROI_gray = cv2.cvtColor(ROI_crop, cv2.COLOR_BGR2GRAY)
        #emotion = emotionDetector_Config.detect_emotions(emotionDetector, ROI_gray)
        print("Emotion {}".format(index_face))

def EyeGaze_Calculation(img, faces, landMarksDetector_Config, landMarksDetector, frame_count):
    print("EyeGaze_Calculation {}".format(faces[0]))
    faces_data = landMarksDetector_Config.detect_landMarks(landMarksDetector,faces,img)
    eye_centers=[]
    for index,POI in enumerate(faces_data):
        eye_corners=POI[2]
        eye_center=eyeGazeEstimation_.getEyePos(eye_corners,img)
        eye_centers.append(eye_center)
        #print(len(eye_centers))
    for index,POI in enumerate(faces_data):
        viewPoint=eyeGazeEstimation_.getCoordFromFace(POI[0],eye_centers[index])
        cv2.line(img,(int(eye_centers[index][0][0]),int(eye_centers[index][0][1])),(int(eye_centers[index][0][0]-viewPoint[0]),int(eye_centers[index][0][1]-viewPoint[1])),(0,255,0),4, -1) # detect landmasks
        
def _Emotion_Calculation(img, faces, faceDetector_Config, emotionDetector_Config, emotionDetector, count_frame):
    for index_face, face in enumerate(faces):
        [x,y,w,h] = face["box"] if faceDetector_Config.type=="mtcnn" else list(face)
        ROI_crop =  img[y:y+h, x:x+w]
        ROI_gray = cv2.cvtColor(ROI_crop, cv2.COLOR_BGR2GRAY)
        emotion = emotionDetector_Config.detect_emotions(emotionDetector, ROI_gray)
        print("Emotion {}".format(count_frame))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        if y > 50:
            cv2.rectangle(img,(x, y-45),(x+len(emotion)*20 -len(emotion)*2 ,y-15),(0,255,255),-1)
            cv2.putText(img,  emotion, (x, y-20), cv2.FONT_HERSHEY_DUPLEX , 1, emotionDetector_Config.emotion_color_dict[emotion] ,2, cv2.LINE_4)
        else: 
            cv2.rectangle(img,(x, y+h+15),(x+len(emotion)*20 -len(emotion)*2 ,y+h+45),(0,255,255),-1)
            cv2.putText(img,  emotion, (x, y+h+40), cv2.FONT_HERSHEY_DUPLEX , 1, emotionDetector_Config.emotion_color_dict[emotion] ,2, cv2.LINE_4)
        return img

def _EyeGaze_Calculation(img, faces, landMarksDetector_Config, landMarksDetector, frame_count):
     # detect landmasks
    faces_data = landMarksDetector_Config.detect_landMarks(landMarksDetector,faces,img)
    eye_centers=[]
    for index,POI in enumerate(faces_data):
        eye_corners=POI[2]
        eye_center=eyeGazeEstimation_.getEyePos(eye_corners,img)
        eye_centers.append(eye_center)

    for index,POI in enumerate(faces_data):
        viewPoint=eyeGazeEstimation_.getCoordFromFace(POI[0],eye_centers[index])
        cv2.line(img,(int(eye_centers[index][0][0]),int(eye_centers[index][0][1])),(int(eye_centers[index][0][0]-viewPoint[0]),int(eye_centers[index][0][1]-viewPoint[1])),(0,255,0),4, -1) # detect landmasks
    print("EyeGaze_Calculation {}".format(frame_count))
    return img

def _EyeGazeEstimation(source_video, techniques):
    config = initialize_Config()
    config.source_video = source_video
    config.type_faceDetection = techniques["Face_Detection"]
    config.type_landMaskDetection = techniques["LandMark_Detection"]
    config.type_eyeGazeEstimation = techniques["EyeGaze_Detection"]
    inputSource = initialize_inputSource(config)
    # Intialize techniques
    # --Face detector
    #detector = dlib.get_frontal_face_detector()
    faceDetector_Config = faceDetection_.FaceDetection()
    faceDetector_Config.process(config)
    faceDetector = faceDetector_Config.intialize()
    # --Landmarks detector
    landMarksDetector_Config = face_landMarkDetection_.Face_LandMarkDetection()
    landMarksDetector_Config.process(config)
    landMarksDetector = landMarksDetector_Config.intialize()
    # --Eyegaze estimatior
    """
    eyeGazeEstimatior_Config = eyeGazeEstimation_.EyeGazeEstimation()
    eyeGazeEstimatior_Config.process(config)
    eyeGazeEstimatior = eyeGazeEstimatior_Config.intialize()
    """
    # --Emotion detector
    print("intialize emotionDetector ...",end=" ")
    emotionDetector_Config = emotionDetection_.EmotionDetection()
    emotionDetector_Config.process(config)
    emotionDetector = emotionDetector_Config.intialize()
    print("OK")
    # time show calculation
    currentTime = 0
    previousTime = 0
    video = cv2.VideoCapture(inputSource.path) # read input Source
    # Run the loop
    count_frame = 0
    while True:
        ret, img = video.read()
        if ret == True:
            count_frame+=1
            if faceDetector_Config.type == "haarcascade_faceDetection":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(gray, 1.3, 5)
                faces = list(faces)
            elif faceDetector_Config.type == "mtcnn":
                faces = faceDetector.detect_faces(img)

            # detect landmasks

            #exec1 = Process(target=Emotion_Calculation, args=(img, faces, faceDetector_Config, emotionDetector_Config, emotionDetector, count_frame)) # pass counter and thread_name into method _counter
            exec2 = Process(target=EyeGaze_Calculation, args=(img, faces, landMarksDetector_Config, landMarksDetector, count_frame))
            execs = [exec2]
            #exec1.start()
            exec2.start()
            for exec in execs:
                exec.join()
            
            
            """
            faces_data = landMarksDetector_Config.detect_landMarks(landMarksDetector,faces,img)
            eye_centers=[]
            for index,POI in enumerate(faces_data):
                eye_corners=POI[2]
                eye_center=eyeGazeEstimation_.getEyePos(eye_corners,img)
                eye_centers.append(eye_center)

            for index,POI in enumerate(faces_data):
                viewPoint=eyeGazeEstimation_.getCoordFromFace(POI[0],eye_centers[index])
                cv2.line(img,(int(eye_centers[index][0][0]),int(eye_centers[index][0][1])),(int(eye_centers[index][0][0]-viewPoint[0]),int(eye_centers[index][0][1]-viewPoint[1])),(0,255,0),4, -1)
            """
            # calculate the processing time
            currentTime = time.time()
            print(currentTime)
            fps = str((currentTime-previousTime))
            previousTime = currentTime
            print("FPS {}|{}".format(fps, inputSource.fps))
            cv2.putText(img,  fps+"|"+str(int(inputSource.fps)), (int(inputSource.width) - 110, 30), cv2.FONT_HERSHEY_DUPLEX , 1, (0,255,0) ,2, cv2.LINE_4)
            cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()


def _counter(counter, process_name):
    while (counter):
        time.sleep(0.01)
        print("{}: {}".format(process_name, counter))
        counter -= 1

def test_muti():
    counter = 5
    exec1 = Process(target=_counter, args=(counter, "khanh thread")) # pass counter and thread_name into method _counter
    exec2 = Process(target=_counter, args=(counter, "ai thread"))
    execs = [exec1, exec2]
    exec1.start()
    exec2.start()

    for exec in execs:
        exec.join()