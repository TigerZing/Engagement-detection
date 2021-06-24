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
import streamlit.components.v1 as components
from PIL import Image
#icon_page = Image.open('icon_page.png')
st.set_page_config(page_title = 'Engagement Detection System', layout="wide")
import pandas as pd
import time
# ------------
import Api_tools.IO_file as IO_file     # import api_tools
from tools import *                     # import tools
from initialization import *            # import initialization
from layout import *                    # import layout
# -Import Objects-
from Objects import config_
from Objects import inputSource_
from Objects import faceDetection_
from Objects import emotionDetection_
from Objects import face_landMarkDetection_
# -Import Algorithms-


def face_detection():
    # Set layout
    chart_plholder, showTime_plholder, status_plholder, info_processing_plholder, fps_plholder, info_algorithm_plholder = layout_FaceDetection()
    info_algorithm_expander = info_algorithm_plholder.beta_expander("**Haarcascade**")
    with info_algorithm_expander:
        st.text(""" This pre-trained model is trained on a dataset including face detection dataset and \nbenchmark (FDDB) [2], WIDER FACE [3], and annotated facial landmarks in the wild (AFLW) benchmark [4].\nFDDB dataset contains the annotations for 5171 faces in a set of 2845 images. \nWIDER FACE dataset consists of 393,703 labeled face bounding boxes in 32,203 images. \nAFLWcontains the facial landmarks annotations for 24 386 faces.""")







    # Run
    status_plholder.text("initialize config...")
    time.sleep(0.5)
    config = initialize_Config()
    status_plholder.text("initialize config...")
    time.sleep(0.5)
    status_plholder.text("initialize input source...")
    time.sleep(0.5)
    inputSource = initialize_inputSource(config)
    status_plholder.text("initialize input source...")


    # --Face detector
    time.sleep(0.5)
    status_plholder.text("initialize face detector...")
    time.sleep(0.5)
    faceDetector_Config, faceDetector = initialize_faceDetector(config)

    # setting for col_chart
    last_rows = [[30]]
    chart = chart_plholder.line_chart()
    chart.add_rows(last_rows)
    time.sleep(0.5)
    status_plholder.text("Processing...")

    # Processing
    video = cv2.VideoCapture(inputSource.path) # read input Source
    currentTime = 0
    previousTime = 0
    # Run the loop
    while True:
        ret, img = video.read()
        if ret == True:
            # Face detection 
            faces = faceDetector_Config.detect_faces(faceDetector, img)
            # Calculate the processing time
            currentTime = time.time()
            fps = str(int(1/(currentTime-previousTime)))
            previousTime = currentTime


            info_processing_plholder.write(f"Deteced: **{len(faces)}** faces")
            fps_plholder.write(f"Frame per second: **{fps}** FPS ")                              
            new_rows = [[len(faces)]]
            chart.add_rows(new_rows)
            last_rows = new_rows
    video.release()

    my_expander = st.beta_expander("List of detected faces")
   

def face_recognition():
    config = initialize_Config()
    inputSource = initialize_inputSource(config)
    # --Face detector
    faceDetector_Config, faceDetector = initialize_faceDetector(config)
    # initialize layout
    col_chart, col_face = st.beta_columns([4,1])
    col_chart.write("Overview")
    col_face.write("Face")
    # setting for col_chart
    last_rows = [[20]]
    chart = col_chart.line_chart(last_rows)
    # Processing
    video = cv2.VideoCapture(inputSource.path) # read input Source
    face_ROI = col_face.empty()
    # Run the loop
    while True:
        ret, img = video.read()
        if ret == True:
            # Face detection 
            faces = faceDetector_Config.detect_faces(faceDetector, img)                              
            for index_face, face in enumerate(faces):
                [x,y,w,h] = face["box"] if faceDetector_Config.type=="mtcnn" else list(face)
                ROI_crop =  img[y:y+h, x:x+w]
                face_ROI.image(ROI_crop, channels="BGR" , width = 120)
            new_rows = [[len(faces)]]
            chart.add_rows(new_rows)
            last_rows = new_rows
    video.release()
    my_expander = st.beta_expander("List of detected faces")
    with my_expander:
        'Hello there!'
        clicked = st.button('Click me!')

def emotion_detection():
    pass

def eye_gaze_estimation():
    pass

# tools

def start_main():
    st.write("Start processing")
    return
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


    # ----Setting-----
    currentTime = 0
    previousTime = 0
    if config.write2File is True:
        _fourcc = cv2.VideoWriter_fourcc("F","M","P","4")
        out = cv2.VideoWriter(config.outputFile_path, _fourcc, float(inputSource.fps), (int(inputSource.width),int(inputSource.height)))
    
    video = cv2.VideoCapture(inputSource.path) # read input Source
    # Run the loop
    while True:
        ret, img = video.read()
        if ret == True:
            # Face detection 
            faces = faceDetector_Config.detect_faces(faceDetector, img)                              
            for index_face, face in enumerate(faces):
                [x,y,w,h] = face["box"] if faceDetector_Config.type=="mtcnn" else list(face)
                ROI_crop =  img[y:y+h, x:x+w]
                ROI_gray = cv2.cvtColor(ROI_crop, cv2.COLOR_BGR2GRAY)
                # Emotion detection 
                emotion = emotionDetector_Config.detect_emotions(emotionDetector, ROI_gray)
                
                # calculate the running time
                currentTime = time.time()
                fps = str(int(1/(currentTime-previousTime)))
                previousTime = currentTime


            if config.drawRunningTime is True:
                draw_runningTime(img, fps, inputSource.width, inputSource.height)

            # Frame Post-Process
            if config.write2File is True:
                out.write(img)
    video.release()
    if config.write2File is True:
        out.release()

# Wenapp components
def user_input_features(max_fps):
    processing_time  = st.sidebar.slider('FPS to process', 1, 3, max_fps)
    data = {'processing_time 1':processing_time}
    return data

if __name__=='__main__':

    # Slidebar
    type_technique = st.sidebar.selectbox("Technique",("Face detection","Emotion detection","Eye gaze estimation","Face recognition"))
    type_algorithm = st.empty()

    if type_technique == "Face detection":
        type_algorithm = st.sidebar.selectbox("Select Face detector | Algorithms | Model",("Haarcascade","MTCNN"))
    elif type_technique == "Emotion detection":
        pass
    elif type_technique == "Eye gaze estimation":
        pass
    elif type_technique == "Face recognition":
        pass
    else:
        print("Technique is unknown !")
    st.sidebar.subheader("User Setting Parameters")
    type_input_source = st.sidebar.selectbox("Input source",("Default","Camera","Streaming video URL"))
    input_source = st.empty()
    if type_input_source == "Camera":
        input_source = st.sidebar.radio('Select device index:', ['device 0', 'device 1'])
    elif type_input_source == "Streaming video URL":
        input_source = st.sidebar.text_input("Enter URL", "https://www.youtube.com/watch?v=yhzwdEhtyCA")
    elif type_input_source == "Default":
        input_source = st.sidebar.write("*Read video [example]*")
    max_fps = 30
    df = user_input_features(max_fps)


    isStart = st.sidebar.button("Start processing")

    # Processing:
    if type_technique == "Face detection":
        if isStart:
            face_detection()
    elif type_technique == "Emotion detection":
        if isStart:
            face_recognition()
    elif type_technique == "Eye gaze estimation":
        if isStart:
            emotion_detection()
    elif type_technique == "Face recognition":
        if isStart:
            face_recognition()
    else:
        print("Technique is unknown !")
        


    #main()