# ---------------------------------------------
# Project: Engagement detection using emotional information
#  written by Nguyen Tan Ho
# 2021/04/15
# ---------------------------------------------
# -Import common packages-
import os
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import sys
import tensorflow as tf
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
import time
import math
from math import atan2,degrees
import random
#icon_page = Image.open('icon_page.png')
st.set_page_config(page_title = 'Engagement Detection System', layout="wide")

# ------------
import Api_tools.IO_file as IO_file     # import api_tools
from tools import *                     # import tools
from initialization import *            # import initialization
from layout import *     
from web_components import *            # import layout
# -Import Objects-
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



def show_info_input(info_input_placeholder, config, inputSource):
    with info_input_placeholder:
        st.write("----")
        if inputSource.type == "video":
            st.write("Source: "+str(osp.basename(inputSource.path)))
        else: 
            st.write("Source: "+str(inputSource.path))
        st.write("Type: "+str(inputSource.type))
        st.write("FPS: "+str(int(inputSource.fps)))
        st.write("Resolution: "+str(int(inputSource.width))+"x"+str(int(inputSource.height)))
        
        if inputSource.type != "online":
            hour, minute_res, second_res = calculate_time_duration(inputSource.fps, inputSource.length)
            st.write("Duration: {}h :{}m :{}s ".format( hour, minute_res, second_res))
        st.write("----")

def initialize_system(input_source, status_running_placeholder):
    status_running_placeholder.write("Extracting configuration file...")
    time.sleep(0)
    config = initialize_Config()
    config.source_video = input_source
    status_running_placeholder.write("Loading input source...")
    time.sleep(0)
    inputSource = initialize_inputSource(config)
    status_running_placeholder.write("Loading controller...")
    time.sleep(0)
    control = initialize_Control(config)
    status_running_placeholder.empty()

    return config, inputSource, control

def initialize_face_detection(status_running_placeholder, config):
    # Intialize techniques
    # --Face detector
    status_running_placeholder.write("Initialize face detection module...")
    time.sleep(0)
    faceDetector_Config, faceDetector = initialize_faceDetector(config)
    status_running_placeholder.empty()
    return faceDetector_Config, faceDetector

def initialize_emotion_detection(status_running_placeholder, config):
    # --Emotion detector
    status_running_placeholder.write("Initialize emotion detection module...")
    time.sleep(0)
    emotionDetector_Config, emotionDetector = initialize_emotionDetector(config)
    status_running_placeholder.empty()
    return emotionDetector_Config, emotionDetector

def initialize_facial_landmark_detection(status_running_placeholder, config):
    # --Facial Landmark detector
    status_running_placeholder.write("Initialize facial landmark detection module...")
    time.sleep(0)
    face_landMarkDetection_Config, face_LandMarkDetector = initialize_face_landMarkDetector(config)
    status_running_placeholder.write("OK!")
    time.sleep(0)
    status_running_placeholder.empty()
    return face_landMarkDetection_Config, face_LandMarkDetector


def frontend():
    # INTERFACE
    # constant settings
    processing_time = None
    flag_maxFPS_process = False
    
    # SLIDERBAR
    # title
    st.sidebar.subheader("Setting Parameters")
    # Input
    type_input_source = st.sidebar.selectbox("Input source",("Default","Streaming video URL"))
    input_source = st.empty()
    if type_input_source == "Camera":
        input_source = st.sidebar.radio('Select device index:', ['device 0', 'device 1'])
    elif type_input_source == "Streaming video URL":
        input_source = st.sidebar.text_input("Enter URL", "https://www.youtube.com/watch?v=mjWxYI4oTTw")
        input_source_value = input_source
    elif type_input_source == "Default":
        input_source_value = "/home/hont/input.mp4"
        input_source = st.sidebar.write("/home/hont/input.mp4")
    # Select the number of frame to be process in one second
    type_input_source = st.sidebar.selectbox("Number of frames to process in one second",("Choose a number", "Max FPS"))
    processing_time_placeholder = st.empty()
    if type_input_source != "Max FPS":
        processing_time_placeholder  = st.sidebar.slider('Number of frames to process in one second', 1, 15, 5)
        processing_time = processing_time_placeholder
        flag_maxFPS_process = False
    else:
        flag_maxFPS_process = True # process all frames in one second
        processing_time_placeholder = st.empty()
    # Select the period of time to analyze
    time_to_analyze_slider  = st.sidebar.slider('Period of time to analyze (second)', 1, 10, 1)

    # check settings
    visualization_checkbox = st.sidebar.checkbox('Visualize image') # visualize image
    export_video_checkbox = st.sidebar.checkbox('Export to output video') # export to output video

    # Start button
    start_button = st.sidebar.button("RUN")
    # place to show information when running system
    progress_bar = st.sidebar.empty()
    status_running_placeholder = st.sidebar.empty()
    
    # info of inputsource
    info_input_placeholder = st.sidebar.beta_container()

    

    # FUCNTION

    if start_button:
        
        # initialize system
        
        config, inputSource, control = initialize_system(input_source_value, status_running_placeholder)
        # adjust config by user parameter settings
        
        config.write2File = export_video_checkbox
        config.drawOnFrame = export_video_checkbox
        if flag_maxFPS_process is True :
            config.time_process = int(inputSource.fps)
        else:
            config.time_process = int(processing_time)


        faceDetector_Config, faceDetector = initialize_face_detection(status_running_placeholder, config)
        emotionDetector_Config, emotionDetector = initialize_emotion_detection(status_running_placeholder ,config)
        face_landMarkDetection_Config, face_LandMarkDetector = initialize_facial_landmark_detection(status_running_placeholder, config)
        # show information of input
        show_info_input(info_input_placeholder, config, inputSource)
        
        
        # DASHBOARD
        st.header("Dashboard System")
        ctop1, ctop2= st.beta_columns((6, 2))
        with ctop1:
            st.write("Engagement detection")
            engagement_bar_chart_value = pd.DataFrame([[0,0,0,0,0]], columns=['strong engagement','medium engagement','high engagement','low engagement','disengagement'])
            engagement_bar_chart  = st.bar_chart(engagement_bar_chart_value)
        with ctop2:
            st.write("Information")
        c1, c2, c3= st.beta_columns((2, 3, 3))
        with c1:
            st.write("Emotion detection")
            # You can call any Streamlit command, including custom components:
            #last_rows = np.random.randn(1, 5)
            emotion_bar_chart  = st.empty()

        with c2:
            st.write("Information")
            st.write("Top Engagement-students")
            img_top_engagement_student_plholder = st.empty()
            st.write("Top Disengagement-students")
            img_top_disengagement_student_plholder = st.empty()
            # You can call any Streamlit command, including custom components:

        with c3:
            #st.write("Concentration detection")
            pass
            #concentrantion_line_value = pd.DataFrame(np.random.randint(10, size = (1,1)), columns=['focused'])
            #concentrantion_line_chart  = st.line_chart(concentrantion_line_value)
            # You can call any Streamlit command, including custom components:

        # Main program
        #if type_input_source!="Camera":
        #    progress_bar.progress(0)

        student_list =[]
        index_time_analyze = -1
        total_analytic_emotion = {"Angry": 0, "Disgusted": 0, "Fearful" :0, "Happy":0, "Neutral":0, "Sad":0, "Surprised":0}
        # ----Setting-----
        currentTime = 0
        previousTime = 0
        if config.write2File is True:
            _fourcc = cv2.VideoWriter_fourcc("F","M","P","4")
            out = cv2.VideoWriter(config.outputFile_path, _fourcc, float(inputSource.fps), (int(inputSource.width),int(inputSource.height)))
        video = cv2.VideoCapture(inputSource.path)
        # Run the loop
        frame_process_jump = int(int(inputSource.fps) / float(config.time_process))
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
                            student._emotion[emotion]+=1

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
                                #print("focus - {}".format(attention))
                            else:
                                if student._status_eyeGaze_count <=3:
                                    attention = "focus"
                                    student._status_eyeGaze_count +=1
                                else:
                                    attention = "distracted"
                                #print("distracted - {}".format(attention))

                        #level_engagement, _ = random.choice(list(student._engagement_level.items()))
                        #student._emotion[random.choice(list(student._emotion.items()))[0]]+=1
                        #student._attention[random.choice(list(student._attention.items()))[0]]+=1
                        #student._engagement_level[random.choice(list(student._engagement_level.items()))[0]]+=1
                        
                        
                        # CALCULATE ENGAGEMENT SCORE
                        engagement_level = engagementDetection_.detect_engagement(emotion, attention)
                        student._engagement_level[engagement_level]+=1
                        #st.write(engagement_level)


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
                


                # analyze results
                time_second_processed = int(control._frame_idx // inputSource.fps)
                if (time_second_processed % time_to_analyze_slider == 0) and ((time_second_processed // time_to_analyze_slider) != index_time_analyze):
                    index_time_analyze = (time_second_processed // time_to_analyze_slider)

                    # visualize engagement analytic results
                    engagement_bar_chart_value = analyze_session_engagement_(student_list)
                    engagement_bar_chart_value_add = pd.DataFrame([engagement_bar_chart_value], columns=['strong engagement','medium engagement','high engagement','low engagement','disengagement'])
                    #print(engagement_bar_chart_value)
                    engagement_bar_chart.add_rows(engagement_bar_chart_value_add)
                    #engagement_bar_chart_value = engagement_bar_chart_value_add

                    # visualize emotion analytic results
                    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
                    labels = "Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"
                    data = analyze_emotion_(student_list, total_analytic_emotion)
                    wedges, texts, autotexts = ax.pie(data, autopct='%1.1f%%',shadow=True, startangle=90)
                    ax.legend(wedges, labels, title="Emotion",loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                    emotion_bar_chart.pyplot(fig)
                    plt.cla()
                    plt.close(fig)

                    # visualize concentrantion analytic results
                    #concentrantion_line_value_add = pd.DataFrame(np.random.randint(10, size = (1,1)), columns = list(concentrantion_line_value))
                    #concentrantion_line_chart.add_rows(concentrantion_line_value_add)
                    #concentrantion_line_value = concentrantion_line_value_add

                    # visualize top engagement students
                    try:
                        list_top_engement_student_dict = analyze_session_top_engagement_(student_list)
                        img_top_engagement_student = []
                        cap_top_engagement_student = []
                        for name_student, face_ROI in list_top_engement_student_dict.items():
                            cap_top_engagement_student.append("21R"+"%03d" %int(name_student))
                            img_top_engagement_student.append(face_ROI)
                        list_top_disengement_student_dict = analyze_session_top_disengagement_(student_list)
                        img_top_disengagement_student = []
                        cap_top_disengagement_student = []
                        for name_student, face_ROI in list_top_disengement_student_dict.items():
                            cap_top_disengagement_student.append("21R"+"%03d" %int(name_student))
                            img_top_disengagement_student.append(face_ROI)
                        img_top_engagement_student_plholder.image(img_top_engagement_student, width=70, caption = cap_top_engagement_student)
                        img_top_disengagement_student_plholder.image(img_top_disengagement_student, width=70, caption = cap_top_disengagement_student)
                    except:
                        print("An exception occurred")

                    

                    #img_top_disengagement_student_plholder.image(img_top_engagement_student, width=70, caption = cap_top_engagement_student)

                    # reset analytic_results of all students
                    reset_analytic_results_all(student_list)
                # Frame Post-Process
                if config.write2File is True:
                    out.write(img_show)
                #if control._frame_idx >=50000 or control._frame_idx > (int(inputSource.length)-int(inputSource.fps)):
                #    break
                if config.showFrame is True:
                    cv2.imshow('frame',img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                control._frame_idx +=1
                #process_value = float(int(control._frame_idx)*100/int(inputSource.length))
                #progress_bar.progress(process_value)
                #status_running_placeholder.write("Processing... {}%".format(round(process_value,2)))

        # Release all 
        video.release()
        if config.write2File is True:
            out.release()
        if config.showFrame is True:
            cv2.destroyAllWindows()



if __name__=='__main__':
    frontend()
