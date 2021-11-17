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
import random
import os
import os.path as osp
import sys
import pathlib 
import tensorflow as tf
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import plotly.express as px
from traitlets.traitlets import default
import plotly.graph_objs as go
from PIL import Image
import face_recognition
import pandas as pd
import time
import math
from math import atan2,degrees
import random
import base64
import zipfile
import glob
import shutil
from datetime import datetime
project_path = pathlib.Path().absolute()
icon_page = Image.open(osp.join(project_path, "media/logo_official.png"))
st.set_page_config(page_title = 'Engagement Detection System',page_icon =icon_page, layout="wide")

# ------------
import Api_tools.IO_file as IO_file     # import api_tools
from tools import *                     # import tools
from initialization import *            # import initialization
from web_components import *            # import layout
from analyze_DB import *
from pages import *

# -Import Database API-
from Database import connect_db
from Database import api_db
from Database import storage
# -Import Objects-
from Objects import user_account_
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
from Objects import frameData_
from Objects import face_recognition_
# -Import Algorithms-

# SET CONSTANT 
font = cv2.FONT_HERSHEY_SIMPLEX
LOGO_IMAGE = osp.join(project_path,"media/logo_login.png")
LABEL_IMAGE = osp.join(project_path,"media/label.png")
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
VIDEOS_PATH = (STREAMLIT_STATIC_PATH / "videos")
st.session_state.videos_path = VIDEOS_PATH
if not VIDEOS_PATH.is_dir():
    VIDEOS_PATH.mkdir()
Output_video_path = osp.join(VIDEOS_PATH, "output.mp4")
coverted_output_video_path = osp.join(VIDEOS_PATH, "coverted_output.mp4")
st.session_state.dict_information = {
                                    "project_path":project_path,
                                    "LOGO_IMAGE":LOGO_IMAGE,
                                    "LABEL_IMAGE":LABEL_IMAGE,
                                    "STREAMLIT_STATIC_PATH":STREAMLIT_STATIC_PATH,
                                    "VIDEOS_PATH":VIDEOS_PATH,
                                    "Output_video_path":Output_video_path,
                                    "coverted_output_video_path":coverted_output_video_path
                                    }


# engagement level color dict 
engagement_level_color = {'strong engagement': (0,255,0), 'high engagement': (50,205,50), 'medium engagement': (60,179,113), 'low engagement': (0,128,0), 'disengagement': (255,0,0)}

# Main components
def show_info_input(info_input_placeholder, config, inputSource):
    info_input_placeholder = st.sidebar.beta_container()
    with info_input_placeholder:
        st.write("----")
        if inputSource.type != "online":
            st.write("Source: "+str(osp.basename(inputSource.path)))
        st.write("Type: "+str(inputSource.type))
        st.write("FPS: "+str(int(inputSource.fps)))
        st.write("Resolution: "+str(int(inputSource.width))+"x"+str(int(inputSource.height)))
        if inputSource.type != "online":
            hour, minute_res, second_res = calculate_time_duration(inputSource.fps, inputSource.length)
            st.write("Length: {}h :{}m :{}s ".format( hour, minute_res, second_res))
            st.session_state.length_video = "{}h :{}m :{}s ".format( hour, minute_res, second_res)
        st.write("----")
        st.session_state.FPS = str(int(inputSource.fps))
        st.session_state.Resolution = str(int(inputSource.width))+"x"+str(int(inputSource.height))
        
def initialize_system(input_source):
    time_delay = 0.3
    status_running_placeholder = st.session_state.status_running_placeholder
    status_running_placeholder.info("Extracting configuration file...")
    time.sleep(time_delay)
    config = initialize_Config()
    config.source_video = input_source
    status_running_placeholder.info("Loading input source...")
    time.sleep(time_delay)
    inputSource = initialize_inputSource(config)
    status_running_placeholder.info("Loading controller...")
    time.sleep(time_delay)
    control = initialize_Control(config)
    status_running_placeholder.empty()

    return config, inputSource, control

def initialize_face_detection(config):
    # Intialize techniques
    # --Face detector
    time_delay = 0.3
    status_running_placeholder = st.session_state.status_running_placeholder
    status_running_placeholder.info("Initialize face detection module...")
    time.sleep(time_delay)
    faceDetector_Config, faceDetector = initialize_faceDetector(config)
    status_running_placeholder.empty()
    return faceDetector_Config, faceDetector

def initialize_emotion_detection(config):
    time_delay = 0.3
    status_running_placeholder = st.session_state.status_running_placeholder
    # --Emotion detector
    status_running_placeholder.info("Initialize emotion detection module...")
    time.sleep(time_delay)
    emotionDetector_Config, emotionDetector = initialize_emotionDetector(config)
    status_running_placeholder.empty()
    return emotionDetector_Config, emotionDetector

def initialize_facial_landmark_detection(config):
    status_running_placeholder = st.session_state.status_running_placeholder
    # --Facial Landmark detector
    status_running_placeholder.info("Initialize facial landmark detection module...")
    time.sleep(0)
    face_landMarkDetection_Config, face_LandMarkDetector = initialize_face_landMarkDetector(config)
    status_running_placeholder.info("OK!")
    time.sleep(0)
    status_running_placeholder.empty()
    return face_landMarkDetection_Config, face_LandMarkDetector

def hide_menu_button(flag):
    if flag == True:
        st.markdown(""" <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style> """, unsafe_allow_html=True)

def hide_stop_button(flag):
    if flag == True:
        st.markdown(".css-htsy04{visibility: hidden; !important}", unsafe_allow_html=True)

# Read database
def load_all_useraccount(conn):
    dataRows = api_db.extract_all_row(conn, "user_account")
    list_user_account = []
    for dataRow in dataRows:
        user_acc = user_account_.User_Account()
        user_acc._convert_dataRow(dataRow)
        list_user_account.append(user_acc)
    return list_user_account

def read_admin_account(list_user_account):
    for user_account in list_user_account:
        if user_account.username == "admin":
            return user_account

def update_date_login(conn, user_account):
    now = datetime.now()
    new_date_login = now.strftime("%d/%m/%Y %H:%M:%S")
    api_db.update_specificValue(conn,"user_account","date_login",new_date_login,"username",user_account.username)

def add_signup_account(conn, user_name, user_password, user_email):
    now = datetime.now()
    new_date_login = now.strftime("%d/%m/%Y %H:%M:%S")
    user_name_db = user_email.split("@")[0]
    values_to_add = "'"+user_name_db+"','"+user_password+"','"+user_email+"','"+user_name+"','"+new_date_login+"'"
    api_db.add_row(conn,"user_account","username, password, email, fullname, date_login",values_to_add)

    list_ = load_all_useraccount(conn)

def title_bar():
    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-weight:700 !important;
            font-size:50px !important;

            color: #0085ad !important;
            padding-left: 50px !important;
        }
        .logo-img {
            width: auto;
            margin-top: -70px !important;
            float:right;
        }
        .line {
            margin-top: -1em;
            height:3px;
        }
        .line_small{
            margin-top: 1em;
            height:3px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" style="width: 20%;" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            <p class="logo-text">ENGAGEMENT DETECTION SYSTEM</p>
        </div>
        <hr class="line">
        """,
        unsafe_allow_html=True
    )


def check_and_add_signup_account(user_name, user_password, user_email):
    database_conn = connect_db.create_connection(osp.join(project_path, st.session_state.config_webservice.database_path))
    # Load user_account database
    list_user_account = load_all_useraccount(database_conn)
    for user_account in list_user_account:
        if user_email == user_account.email:
            return False
    add_signup_account(database_conn, user_name, user_password, user_email)
    database_conn.close()
    return True


def process_signup():
    user_email = st.session_state.username
    user_name = st.session_state.userFullname
    user_password = st.session_state.password
    if user_name != "" and user_email != "" and user_password != "":
        if check_and_add_signup_account(user_name, user_password, user_email) is True and user_email.find("@") != -1:
        # check ok or not:
            st.session_state.login_gate = True
            st.session_state.isSignUp = False
            st.session_state.loginNoti.success('Your sign up was successful!')
            time.sleep(0.5)
            st.session_state.loginNoti.empty()
            
        else:
            if user_email.find("@") == -1:
                st.session_state.loginNoti.error('This email address is not valid')
            else:
                st.session_state.loginNoti.error('This email address has been used already! Please enter another one!')
            time.sleep(2)
            st.session_state.loginNoti.empty()
    else:
        st.session_state.loginNoti.error('Please fill in the blank')
        time.sleep(1)
        st.session_state.loginNoti.empty()

def process_login():
    database_conn = connect_db.create_connection(osp.join(project_path, st.session_state.config_webservice.database_path))
    # Load user_account database
    list_user_account = load_all_useraccount(database_conn)
    user_name = st.session_state.username
    user_password = st.session_state.password
    for user_account in list_user_account:
        if (user_name == user_account.email or user_name == user_account.username) and user_password == user_account.password:

            # check ok or not:
            st.session_state.login_gate = True
            st.session_state.isSignUp = False
            st.session_state.user_data_storage = osp.join(project_path, user_account.user_data_storage)
            st.session_state.loginNoti.success('Login successful!')
            time.sleep(0.5)
            st.session_state.loginNoti.empty()
            st.session_state.list_user_account = None
            update_date_login(database_conn, user_account)
            database_conn.close()
            return
    st.session_state.loginNoti.error('Incorrect!')
    time.sleep(1)
    st.session_state.loginNoti.empty()


def routine_to_Signup():
    st.session_state.isSignUp = True


def SignUp_page():
    
    st.markdown('<br>', unsafe_allow_html=True)
    cEmpty1, cMain, cEmpty2 = st.beta_columns((5, 3, 5))
    with cMain:
        st.markdown(f"""<img src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">""", unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        user_name = st.text_input("Full name", value="Hosei Takaro")
        user_email = st.text_input("Email")
        user_password = st.text_input("Password ", type="password")

        st.session_state.username = user_email
        st.session_state.userFullname = user_name
        st.session_state.password = user_password
        submitButton_pressed = st.button("Submit", on_click=process_signup)
        loginNoti = st.empty()
        st.session_state.loginNoti = loginNoti
    


def LogIn_page():
    #title_bar()
    st.markdown('<br><br><br>', unsafe_allow_html=True)
    cEmpty1, cMain, cEmpty2 = st.beta_columns((5, 3, 5))
    with cMain:
        st.markdown(f"""<img src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">""", unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        username = st.text_input("Username", value=st.session_state.username)
        st.session_state.username = username
        password = st.text_input("Password", type="password", value=st.session_state.password)
        st.session_state.password = password
        loginButton_pressed = st.button("Login", on_click=process_login)
        #st.write("Don't have an account yet? sign up here!")
        #signUp_checked = st.button("Sign up", help="Sign up to use the system", on_click=routine_to_Signup)
        loginNoti = st.empty()
        st.session_state.loginNoti = loginNoti


def switch_state_processing():
    if st.session_state.isRunning is False:
        st.session_state.isFirstTime = False
        st.session_state.isRunning = True
    else :
        st.session_state.isRunning = False
        st.session_state.status_running_placeholder.error("Stopped!")
        if "out_video" in st.session_state:
            st.session_state.status_running_placeholder.warning("Rendering video ...!")
            st.session_state.out_video.release()
            cmd_str = "ffmpeg -y -i "+Output_video_path+" -vcodec libx264 "+ coverted_output_video_path
            os.system(cmd_str)
        st.session_state.video.release()
        time.sleep(0.2)
        st.session_state.status_running_placeholder.empty()


def Input_page_for_user():
    st.session_state.user_type = "user"
    input_page_container = st.sidebar.beta_container()
    #input_page_container.write("# Welcome: {}".format(st.session_state.userFullname))
    input_page_container.subheader("Settings")

    now = datetime.now()
    random_value = now.strftime("%d%m%Y%H%M%S")
    specific_date = now.strftime("%H:%M %Y/%m/%d")

    # Input
    input_source_type =  input_page_container.selectbox("Source", ('Enter an streaming URL','Upload a video'))
    input_source = None
    if input_source_type == 'Upload a video':
        input_source_file = input_page_container.file_uploader("Recorded video",type=['mp4'])
        if input_source_file is not None:
            # clear all UploadFiles folder
            upload_files_temp = glob.glob(os.path.join(project_path, "UploadFiles","*"))
            for file_ in upload_files_temp:
                os.remove(file_)
            input_source = os.path.join(project_path, "UploadFiles", str(random_value) +"_"+input_source_file.name)
            with open(input_source, "wb") as f:
                f.write(input_source_file.getbuffer())
    else:
        input_source = input_page_container.text_input("Enter the URL of video or streaming", "/home/hont/Engagement_Detection/UploadFiles/video_input.mp4")

    course_id = input_page_container.text_input("Course ID", "ID_"+str(random_value))
    course_name = input_page_container.text_input("Course name", "NAME_"+str(random_value)[-2:])
    lecture_id = input_page_container.text_input("Lecture ID", "LESSON-"+str(random_value)[-2:])
    lecturer_name = input_page_container.text_input("Lecturer name", "USER")
    email_address = input_page_container.text_input("Email address", "username@gmail.com")
    total_students = input_page_container.text_input("Total students","5")
    time_begin = input_page_container.text_input("Class time",specific_date)
    # Define the number of intervals in video
    defineInterval_type =  input_page_container.selectbox("Define the number of intervals to process video", ( 'Set automatically', 'Set the number of intervals',))
    num_interval = input_page_container.slider("", min_value=1, max_value=20, value=10) if defineInterval_type =='Set the number of intervals' else -1 # -1 mean auto set number of interval to process
    # upload database for face recognition
    database_file = input_page_container.file_uploader("Database for face recognition (Optional)",type=['zip'])
    if database_file is not None:
        if "Face_Upload" not in st.session_state:
            st.session_state.Face_Upload = True
        file_details = {"FileName":database_file.name,"FileType":database_file.type,"FileSize":database_file.size}
        input_page_container.write(file_details)
        uploadFile_path = os.path.join(project_path, "UploadFiles", database_file.name)
        st.session_state.uploadFile_path = uploadFile_path
        with open(uploadFile_path, "wb") as f:
            f.write(database_file.getbuffer())
    else:
        if "Face_Upload" not in st.session_state:
            st.session_state.Face_Upload = False

    # select which method to mapping
    mappingMethod_type =  input_page_container.selectbox("Select method to map the features", ( 'Pseudo mapping', 'KES mapping',))

    # check settings
    send_email_checkbox = input_page_container.checkbox('Send a Summary Report to my email', True) # visualize image
    export_video_checkbox = input_page_container.checkbox('Write to ouput video ', True) # export to output video
    #debug_mode = input_page_container.checkbox('Debug mode (process 1 time/s)') # debug mode
    debug_mode = True
    show_frame = input_page_container.checkbox('Show analyzed frame') # debug mode
    
    # Start button
    if st.session_state.isRunning is False:
        if st.session_state.isFirstTime !=False:
            start_button = st.sidebar.button("START", on_click=switch_state_processing)
    else:
        start_button = st.sidebar.button("STOP", on_click=switch_state_processing)
    status_running_placeholder = input_page_container.empty()
    if "status_running_placeholder" not in st.session_state:
        st.session_state.status_running_placeholder = status_running_placeholder
    info_input_placeholder = input_page_container.empty()

    input_page_data = {"input_page_container": input_page_container ,
                        "input_source":input_source, 
                        "course_id":course_id, 
                        "course_name":course_name,
                        "lecture_id":lecture_id,
                        "lecturer_name":lecturer_name,
                        "email_address":email_address,
                        "time_begin":time_begin, 
                        "total_students":total_students, 
                        "mappingMethod_type":mappingMethod_type,
                        "num_interval":num_interval,
                        "database_file":database_file, 
                        "send_email_checkbox":send_email_checkbox,
                        "export_video_checkbox":export_video_checkbox, 
                        "debug_mode":debug_mode, 
                        "info_input_placeholder":info_input_placeholder, 
                        "show_frame":show_frame}
        
    return input_page_data

def Input_page_for_admin():
    st.session_state.user_type = "admin"
    input_page_container = st.sidebar.beta_container()
    input_page_container.write("# ADMINISTRATOR")
    input_page_container.subheader("Settings")

    now = datetime.now()
    random_value = now.strftime("%d%m%Y%H%M%S")
    specific_date = now.strftime("%H:%M %Y/%m/%d")

    # Input
    input_source = input_page_container.text_input("Enter the URL of video or streaming", "/home/hont/input_video/input0.mp4")
    # Select the number of frame to be process in one second
    numFrame_processed = input_page_container.slider("Number of frame to be processed in one second", min_value=1, max_value=10, value=1)
    # Define the number of intervals in video
    defineInterval_type =  input_page_container.selectbox("Define the number of intervals to process video", ( 'Set automatically', 'Set the number of intervals',))
    num_interval = input_page_container.slider("", min_value=1, max_value=20, value=10) if defineInterval_type =='Set the number of intervals' else -1 # -1 mean auto set number of interval to process
    # upload database for face recognition
    database_file = input_page_container.file_uploader("Database for face recognition (Optional)",type=['zip'])
    if database_file is not None:
        file_details = {"FileName":database_file.name,"FileType":database_file.type,"FileSize":database_file.size}
        input_page_container.write(file_details)
        uploadFile_path = os.path.join(project_path, "UploadFiles", database_file.name)
        st.session_state.uploadFile_path = uploadFile_path
        with open(uploadFile_path, "wb") as f:
            f.write(database_file.getbuffer())

    # select which method to mapping
    mappingMethod_type =  input_page_container.selectbox("Select method to map the features", ( 'Pseudo mapping', 'KES mapping',))

    # check settings
    send_email_checkbox = input_page_container.checkbox('Send a Summary Report to my email', True) # visualize image
    export_video_checkbox = input_page_container.checkbox('Write to ouput video ', True) # export to output video
    debug_mode = input_page_container.checkbox('Debug mode (process 1 time/s)') # debug mode
    show_frame = input_page_container.checkbox('Show analyzed frame') # debug mode
    
    # Start button
    if st.session_state.isRunning is False:
        start_button = st.sidebar.button("START", on_click=switch_state_processing)
    else:
        start_button = st.sidebar.button("STOP", on_click=switch_state_processing)
    status_running_placeholder = input_page_container.empty()
    if "status_running_placeholder" not in st.session_state:
        st.session_state.status_running_placeholder = status_running_placeholder
    info_input_placeholder = input_page_container.empty()

    input_page_data = {"input_page_container": input_page_container , 
                        "numFrame_processed":numFrame_processed, 
                        "input_source":input_source, 
                        "course_id": "ID_"+str(random_value), 
                        "course_name": "NAME_"+str(random_value)[-2:], 
                        "lecture_id":"LESSON-"+str(random_value)[-2:],
                        "lecturer_name":"USER",
                        "email_address":"nguyentanhoit@gmail.com",
                        "time_begin":specific_date, 
                        "total_students":"5",  
                        "num_interval":num_interval, 
                        "mappingMethod_type":mappingMethod_type,
                        "database_file":database_file, 
                        "send_email_checkbox":send_email_checkbox,
                        "export_video_checkbox":export_video_checkbox, 
                        "debug_mode":debug_mode, 
                        "info_input_placeholder":info_input_placeholder, 
                        "show_frame":show_frame}

    return input_page_data

def Summary_page(input_page_data):
    max_value_num_Detected = 1
    numDetected_list = {"default":0}

    print("st.session_state.Face_Upload {}".format(st.session_state.Face_Upload))

    if "uploadFile_path" not in st.session_state :
        for stu in st.session_state.student_list:
            numDetected_list.update({stu._name:stu._num_Detected})
            print("name: {} {}".format(stu._name,stu._num_Detected))
            if stu._num_Detected >= max_value_num_Detected:
                max_value_num_Detected = stu._num_Detected
    
    

    # Set CSS for this page 
    color_discrete_map_for_engagement= {'strong engagement': '#3c6dc5', 'medium engagement': '#3c6dc5', 'high engagement': '#3c6dc5', 'low engagement': '#3c6dc5', 'disengagement': '#bf360c'}
    CI_color_discrete_map = {'focused':'#8FBBD9', 'distracted':'#FFBF86'}
    emotion_color_discrete_map= {'Angry': '#EE6666', 'Disgusted': '#5470C6', 'Fearful': '#73C0DE', 'Happy': '#FAC858', 'Neutral': '#3BA272', 'Sad':'#91CC75', 'Surprised':'#FC8452'}
    # st.markdown(
    #     """
    #     <style>
    #     .block-container{
    #         border-style: double;
    #         border-width: 10px;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    # analyze database
    database_conn = connect_db.create_connection(osp.join(project_path, st.session_state.config_webservice.database_path))
    stack_barchart, CI_area_chart, fig_Emotion, fig_Engagement, fig_CI, class_summary, student_summary, student_arr, disengagement_dict =  analyze_db_main(database_conn, 10, 25)
    database_conn.close()
    #st.write(class_summary)
    st.markdown(f"""<div style="text-align: center;"><img src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}"></div>""", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    #st.session_state.status_running_placeholder.empty()
    st.markdown('<h1 style="color:#be0027; text-align:center;"> CLASS SUMMARY </h1>', unsafe_allow_html=True)
    now = datetime.now()
    new_date_login = now.strftime("%d/%m/%Y %H:%M:%S")
    st.markdown('<p  style="color:#050f2c; text-align:center;"> <span> <strong>Date: </strong> {}</span> </p>'.format(str(new_date_login)), unsafe_allow_html=True) 
    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#be0027; text-align:left;"><strong> 1. OVERALL </strong></h2><hr class="line_small">', unsafe_allow_html=True)
    
    title_info1, _, title_info2 = st.beta_columns((21, 1, 31))
    with title_info1:
        st.markdown("<h3 style='color:#0085ad; text-align:center;'> Basic infomation </h3><br>", unsafe_allow_html=True)
    with title_info2:
        st.markdown("<h3 style='color:#0085ad; text-align:center;'> Overview analytics </h3><br>", unsafe_allow_html=True)



    info0, _,info1, _, info2, _, info3, _, info4= st.beta_columns((10, 2, 12, 2, 10, 2, 8, 2, 14))

    if input_page_data["course_id"] =="":
        input_page_data["course_id"] = "N/A"
    if input_page_data["course_name"] =="":
        input_page_data["course_name"] = "N/A"
    if input_page_data["total_students"] =="":
        input_page_data["total_students"] = "N/A"
    if input_page_data["time_begin"] =="":
        input_page_data["time_begin"] = "N/A"
    if input_page_data["num_interval"] == "":
        input_page_data["num_interval"] = "N/A"
    elif input_page_data["num_interval"] == -1:
        input_page_data["num_interval"] = "auto"
    if "Resolution" not in st.session_state:
        st.session_state.Resolution = "N/A"
    if "FPS" not in st.session_state:
        st.session_state.FPS = "N/A"
    if "length_video" not in st.session_state:
        st.session_state.length_video = "N/A"
    
    with info0:
        st.markdown("""
        <table cellspacing="0" cellpadding="0">
        <tr>
            <th>Lecturer</th>
            <td>{}</td>
        </tr>
        <tr>
            <th>Email address</th>
            <td>{}</td>
        </tr>
        </table>
        """.format(input_page_data["lecturer_name"], input_page_data["email_address"]),unsafe_allow_html=True)
    with info1:
        st.markdown("""
        <table cellspacing="0" cellpadding="0">
        <tr>
            <th>Course ID</th>
            <td>{}</td>
        </tr>
        <tr>
            <th>Course Name</th>
            <td>{}</td>
        </tr>
        <tr>
            <th>Lecture ID</th>
            <td>{}</td>
        </tr>
        </table>
        """.format(input_page_data["course_id"], input_page_data["course_name"],input_page_data["lecture_id"]),unsafe_allow_html=True)
    with info2:
        st.markdown("""
        <table cellspacing="0" cellpadding="0">
        <tr>
            <th>Length of video</th>
            <td>{}</td>
        </tr>
        <tr>
            <th>FPS</th>
            <td>{}</td>
        </tr>
        <tr>
            <th>Resolution</th>
            <td>{}</td>
        </tr>
        </table>
        """.format(st.session_state.length_video, st.session_state.FPS, st.session_state.Resolution),unsafe_allow_html=True)
    with info3:
        st.markdown("""
        <table cellspacing="0" cellpadding="0">
        <tr>
            <th>Total students</th>
            <td>{}</td>
        </tr>
        <tr>
            <th>Num_intervals</th>
            <td>{}</td>
        </tr>
        <tr>
            <th>Class time</th>
            <td>{}</td>
        </tr>
        </table>
        """.format(input_page_data["total_students"], input_page_data["num_interval"], input_page_data["time_begin"]), unsafe_allow_html=True)
    with info4:
        st.markdown("""
        <table cellspacing="0" cellpadding="0">
        <tr>
            <th>Engagement level</th>
            <th style="color:{}">{} <span style="color:#262730;">({}%)</span> </td>
        </tr>
        <tr>
            <th>Emotion type</th>
            <th style="color:{};">{} <span style="color:#262730;">({}%)</span></th>
        </tr>
        <tr>
            <th>Concentration type</th>
            <th style="color:{};">{} <span style="color:#262730;">({}%)</span></th>
        </tr>
        </table>
        """.format(color_discrete_map_for_engagement[class_summary["Engagement_level"]["top"]] ,class_summary["Engagement_level"]["top"], int(class_summary["Engagement_level"]["Percentage"]), emotion_color_discrete_map[class_summary["Emotion"]["top"]], class_summary["Emotion"]["top"], int(class_summary["Emotion"]["Percentage"]), CI_color_discrete_map[class_summary["Attention"]["top"]], class_summary["Attention"]["top"], int(class_summary["Attention"]["Percentage"])),unsafe_allow_html=True)
        #st.markdown('<p> <span> <strong>Engagement level</strong></span>            :  {} ({}%) </p>'.format(class_summary["Engagement_level"]["top"], int(class_summary["Engagement_level"]["Percentage"])), unsafe_allow_html=True)
        #st.markdown('<p> <span> <strong>Emotion type</strong></span>           :  {} ({}%)</p>'.format(class_summary["Emotion"]["top"], int(class_summary["Emotion"]["Percentage"])), unsafe_allow_html=True)
        #st.markdown('<p> <span> <strong>Concentration type</strong></span>             :  {} ({}%)</p>'.format(class_summary["Attention"]["top"], int(class_summary["Attention"]["Percentage"])), unsafe_allow_html=True)

    st.markdown('<br><br>', unsafe_allow_html=True)
    ctop3, _, ctop1= st.beta_columns((15, 1, 15))
 
    with ctop1:
        st.markdown("<h3 style='color:#0085ad; text-align:center;'> CONCENTRATION VISUALIZATION </h3>", unsafe_allow_html=True)
        concentration_chart_placeholder = st.empty()
        # if input_page_data["course_id"] =="":
        #     input_page_data["course_id"] = "N/A"
        # st.markdown('<p> <span style="color:#be0027;"> <strong>Class ID</strong></span>            :  {} </p>'.format(input_page_data["course_id"]), unsafe_allow_html=True)
        # if input_page_data["course_name"] =="":
        #     input_page_data["course_name"] = "N/A"
        # st.markdown('<p> <span style="color:#be0027;"> <strong>Class Name</strong></span>           :  {} </p>'.format(input_page_data["course_name"]), unsafe_allow_html=True)
        # st.markdown('<p> <span style="color:#be0027;"> <strong>Lecturer</strong></span>             :  {} </p>'.format(str(st.session_state.userFullname)), unsafe_allow_html=True)
        # if input_page_data["total_students"] =="":
        #     input_page_data["total_students"] = "N/A"
        # st.markdown('<p> <span style="color:#be0027;"> <strong>Total of student</strong></span>     :  {} </p>'.format(input_page_data["total_students"]), unsafe_allow_html=True)
        # if input_page_data["time_begin"] =="":
        #     input_page_data["time_begin"] = "N/A"
        # st.markdown('<p> <span style="color:#be0027;"> <strong>Time</strong></span>                 :  {}</p>'.format(input_page_data["time_begin"]), unsafe_allow_html=True)
        concentration_chart_placeholder.plotly_chart(CI_area_chart, use_container_width=True)
    with ctop3:
        # placeholder 
        st.markdown("<h3 style='color:#0085ad; text-align:center;'> ENGAGEMENT VISUALIZATION </h3>", unsafe_allow_html=True)
        engagement_chart_placeholder = st.empty()
        #             # settings
        # fig = px.bar(st.session_state.data_Engagement, 
        #             template = "none",
        #             x =  st.session_state.timeline_data_Engagement,
        #             y = [c for c in st.session_state.data_Engagement.columns],
        #             color_discrete_map={'strong engagement': '#C7DBFF', 'medium engagement': '#6FA4FF', 'high engagement': '#8DB6FF', 'low engagement': '#5291FF', 'disengagement': '#bf360c'},
        #             labels={'value':'Number of student', 'x':'Timeline', 'variable':'Engagement level'},
        #             height=380
        #             )
        # fig.update_layout(barmode='stack',font=dict(size=16), margin=dict(t=10),plot_bgcolor='rgba(0,0,0,0)',hovermode="x")
        # visualize figure
        engagement_chart_placeholder.plotly_chart(stack_barchart, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    mid1, _, mid2, _, mid3 = st.beta_columns((12, 1, 10, 1, 12))
    
    with mid2:
        st.markdown("<h3 style='color:#0085ad; text-align:center;'> EMOTION </h3>", unsafe_allow_html=True)
        # placeholder
        emotion_chart_placeholder = st.empty()
        # data_EmotionLabel = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        # #data_EmotionScore = [12,18,30,10,5,15,10]
        # # settings
        # data = go.Pie(labels = data_EmotionLabel, values = st.session_state.data_EmotionScore)
        # layout = go.Layout(font=dict(size=14),autosize=False,width=400,height=400, 
        #         xaxis= go.layout.XAxis(linecolor = 'black', linewidth = 1, mirror = True),
        #         yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 3, mirror = True),
        #         margin=go.layout.Margin(l=50, r=50, b=100, t=10,pad = 4))
        # fig = go.Figure(data = [data] ,layout=layout)
    
        # colors = ['#EE6666', '#5470C6','#73C0DE','#FAC858','#3BA272','#91CC75','#FC8452']
        # fig.update_traces(hoverinfo='label+value', textinfo='label+percent',marker=dict(colors=colors))
        # visualize figure
        emotion_chart_placeholder.plotly_chart(fig_Emotion, use_container_width=False)
    
    with mid1:
        st.markdown("<h3 style='color:#0085ad; text-align:center;'> ENGAGEMENT DISTRIBUTION </h3>", unsafe_allow_html=True)
        engagement_piechart_placeholder = st.empty()
        # # settings
        # fig = px.area(st.session_state.data_ConcentrationData, 
        #             template = "none",
        #             x = st.session_state.data_ConcentrationData.index,
        #             y = [c for c in st.session_state.data_ConcentrationData.columns], 
        #             height=380,
        #             labels={'value':'Number of student', 'x':'Timeline', 'variable':'Concentration level'},
        #             )
        # fig.update_layout(barmode='stack',font=dict(size=16), margin=dict(t=10),plot_bgcolor='rgba(0,0,0,0)')
        # # visualize figure
        engagement_piechart_placeholder.plotly_chart(fig_Engagement, use_container_width=True)
    #st.markdown("<h3 style='color:#0085ad; text-align:center;'> OUPUT VISUALIZATION </h3><br>", unsafe_allow_html=True)
    with mid3:
        st.markdown("<h3 style='color:#0085ad; text-align:center;'> CONCENTRATION VISUALIZATION </h3>", unsafe_allow_html=True)
        concentration_piechart_placeholder = st.empty()
        # # settings
        # fig = px.area(st.session_state.data_ConcentrationData, 
        #             template = "none",
        #             x = st.session_state.data_ConcentrationData.index,
        #             y = [c for c in st.session_state.data_ConcentrationData.columns], 
        #             height=380,
        #             labels={'value':'Number of student', 'x':'Timeline', 'variable':'Concentration level'},
        #             )
        # fig.update_layout(barmode='stack',font=dict(size=16), margin=dict(t=10),plot_bgcolor='rgba(0,0,0,0)')
        # # visualize figure
        concentration_piechart_placeholder.plotly_chart(fig_CI, use_container_width=True)
    #st.markdown("<h3 style='color:#0085ad; text-align:center;'> OUPUT VISUALIZATION </h3><br>", unsafe_allow_html=True)
    output_video_expander = st.beta_expander(" ► OUPUT VIDEO")
    if st.session_state.config_webservice.write2File is True:
        video_file = open(coverted_output_video_path, 'rb')
        video_bytes = video_file.read()
        output_video_expander.video(video_bytes)
        image_ = Image.open(LABEL_IMAGE)
        output_video_expander.image(image_, width=900)

    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#be0027'; text-align:left;'><strong> 2. INDIVIDUAL ENGAGEMENT</strong></h2><hr class='line_small'>", unsafe_allow_html=True)
    isShowAll = st.checkbox("Show all chart visualization")
    st.markdown('<br>', unsafe_allow_html=True)


    for student_info in student_arr:
        if "uploadFile_path" not in st.session_state and numDetected_list[student_info["student_id"]]<(1/2*max_value_num_Detected):
            continue
        studentFace_column, studentInfo_column, studentResult_column = st.beta_columns((1, 3, 7))
        with studentFace_column:
            #face_visualization = Image.open('/home/hont/Engagement_Detection/media/student.jpg')
            face_visualization = st.session_state.imagesDatabase[student_info["student_id"]]
            st.image(face_visualization, width=120)
            st.markdown('<p> <span> <strong>{}</strong></span></p>'.format(student_info["student_id"]), unsafe_allow_html=True)
        with studentInfo_column:


            engagement_type = student_summary[student_info["student_id"]]["Engagement_level"]["top"]
            engagement_percentage = student_summary[student_info["student_id"]]["Engagement_level"]["Percentage"]
            color_type = color_discrete_map_for_engagement[engagement_type]

            emotion_type = student_summary[student_info["student_id"]]["Emotion"]["top"]
            emotion_percentage = student_summary[student_info["student_id"]]["Emotion"]["Percentage"]

            CI_type = student_summary[student_info["student_id"]]["Attention"]["top"]
            CI_percentage = student_summary[student_info["student_id"]]["Attention"]["Percentage"]



            first_previous_time = None
            first_current_time = None

            second_previous_time = None
            second_current_time = None

            if student_info["student_id"] in disengagement_dict:

                first_previous_time = disengagement_dict[student_info["student_id"]][0]["previous_time"]
                first_current_time = disengagement_dict[student_info["student_id"]][0]['current_time']
                if len(disengagement_dict[student_info["student_id"]]) >1:
                    second_previous_time = disengagement_dict[student_info["student_id"]][1]["previous_time"]
                    second_current_time = disengagement_dict[student_info["student_id"]][1]['current_time']
            
            st.markdown("""
            <table cellspacing="0" cellpadding="0" width ="100%">
            <tr>
                <th width ="200px">Overall Engagement</th>
                <th style="color:{};">{} ({}%)</th>
            </tr>
            <tr>
                <th>Overall Emotion</th>
                <td>{} ({}%)</td>
            </tr>
            <tr>
                <th>Overall Concentration</th>
                <td>{} ({}%)</td>
            </tr>
            <tr>
                <th>Most Distracted analytic</th>
                <td><table><tr><th>1st Distracted time</th><td>{} - {}</td></tr><tr><th>2nd Distracted time</th><td>{} - {}</td></tr></table></td>
            </tr>
            </table>
            """.format(color_type, engagement_type, int(engagement_percentage), emotion_type, int(emotion_percentage), CI_type, int(CI_percentage),first_previous_time ,first_current_time, second_previous_time, second_current_time), unsafe_allow_html=True)
            st.markdown('<br><br>', unsafe_allow_html=True)
        studentResult_column = studentResult_column.beta_expander("STUDENT'S ENGAGEMENT TIMELINE VISUALIZATION", expanded=isShowAll)
        with studentResult_column:
            engagement_chart_placeholder = st.empty()
            # visualize figure
            engagement_chart_placeholder.plotly_chart(student_info["figure"], use_container_width=True)  
     
def Main_page(input_page_data):
    # # Set CSS for this page 
    # st.markdown(
    #     """
    #     <style>
    #     .plot-container {
    #         border-style: solid;
    #         border-width: 1px;
    #         height:410px
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )
    title_bar()
    #Summary_page()
    if st.session_state.isRunning is True:

        ctop= st.beta_container()
        # set some dataset
        data_Engagement = pd.DataFrame()
        data_EmotionLabel = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        data_EmotionScore = [0,0,0,0,0,0,0]
        data_ConcentrationData = pd.DataFrame()
        with ctop:
            st.markdown('<h3 style="color:#008374"> Engagement Visualization </h3>', unsafe_allow_html=True)
            # placeholder 
            engagement_chart_placeholder = st.empty()

        # set second row (including 3 columns)
        cmid1, _, cmid2, _, cmid3= st.beta_columns((5,1, 10, 1,5))
        with cmid1:
            st.markdown('<h3 style="color:#008374"> Emotion Visualization </h3>', unsafe_allow_html=True)
            # placeholder
            emotion_chart_placeholder = st.empty()

        with cmid2:
            st.markdown('<h3 style="color:#008374"> Concentration Visualization </h3>', unsafe_allow_html=True)
            # placeholder
            concentration_chart_placeholder = st.empty()

        with cmid3:
            show_frame_plholder = None
            if input_page_data["show_frame"] is False:
                st.markdown('<h3 style="color:#008374"> Student Visualization </h3>', unsafe_allow_html=True)
                st.markdown('<p style="color:#00a98f"> Top Engagement-students </p>', unsafe_allow_html=True)
                img_top_engagement_student_plholder = st.empty()
                st.markdown('<p style="color:#be0027"> Top Disengagement-students </p>', unsafe_allow_html=True)
                img_top_disengagement_student_plholder = st.empty()
            else:
                st.markdown('<h3 style="color:#008374"> Frame analysis </h3>', unsafe_allow_html=True)
                show_frame_plholder = st.empty()
        
        # MAIN PROCESSING
        # initialize system
        config, inputSource, control = initialize_system(input_page_data["input_source"])
        st.session_state.config = config
        # adjust config by user parameter settings
        st.session_state.config_webservice.write2File = input_page_data["export_video_checkbox"]
        config.time_process = 1

        faceDetector_Config, faceDetector = initialize_face_detection(config)
        emotionDetector_Config, emotionDetector = initialize_emotion_detection(config)
        face_landMarkDetection_Config, face_LandMarkDetector = initialize_facial_landmark_detection(config)
        
        # show the message 
        st.session_state.status_running_placeholder.warning("Running...")

        # show information of input
        show_info_input(input_page_data["info_input_placeholder"], config, inputSource)
        student_list =[]
        st.session_state.student_list = student_list
        timeline_ = []
        index_time_analyze = -1
        total_analytic_emotion = {"Angry": 0, "Disgusted": 0, "Fearful" :0, "Happy":0, "Neutral":0, "Sad":0, "Surprised":0}
        # ----Setting-----
        currentTime = 0
        previousTime = 0
        if st.session_state.config_webservice.write2File is True:
            _fourcc = cv2.VideoWriter_fourcc("F","M","P","4")
            #if 'out_video' not in st.session_state:
            st.session_state.out_video = cv2.VideoWriter(Output_video_path, _fourcc, float(inputSource.fps), (int(inputSource.width),int(inputSource.height)))
            
        video = cv2.VideoCapture(inputSource.path)
        st.session_state.video = video
        # Run the loop
        frame_process_jump = int(int(inputSource.fps) / float(config.time_process))
        time_per_second = 1 if input_page_data["debug_mode"] ==True else 60
        time_to_visualize = 5 # if source is streaming or online video then time = 4 mins
        if inputSource.type == "video": 
            if st.session_state.user_type == "admin" and input_page_data["num_interval"]!=-1:
                time_to_visualize = (int(inputSource.length) // int(input_page_data["num_interval"]))//int(inputSource.fps)//60
            elif st.session_state.user_type == "user" and input_page_data["num_interval"]==-1:
                time_to_visualize = 5
            else:
                time_to_visualize = (int(inputSource.length) // int(input_page_data["num_interval"]))//int(inputSource.fps)//60

        if time_to_visualize == 0:
            time_to_visualize = 1
        print("time_to_visualize: {}".format(time_to_visualize))

        length_hour = None 
        if inputSource.type != "online":
            hour, minute_res, second_res = calculate_time_duration(inputSource.fps, inputSource.length)
            length_hour = "{}h :{}m :{}s ".format( str(hour), str(minute_res), str(second_res))
        else:
            length_hour = "N/A"
            if "info_process" in st.session_state:
                st.session_state.info_process = {"time_to_visualize":time_to_visualize, "length_hour" : length_hour, "fps": str(int(inputSource.fps))}



        # reset tables?
        if st.session_state.config_webservice.reset_analytic_tables is True:
            reset_tables(config, st.session_state.config_webservice)


        # create user_data storage for this session
        list_session = glob.glob(osp.join(st.session_state.user_data_storage,"*"))
        if len(list_session) >=5: # clear all sessions if exceeding 5 sessions
            for f in list_session:
                shutil.rmtree(f)

        # create session_data
        now = datetime.now()
        random_value = now.strftime("%d%m%Y_%H%M%S")
        session_path = osp.join(st.session_state.user_data_storage, random_value)
        os.mkdir(session_path)
        st.session_state.session_storage = session_path

        # extract database for face_recognition
        encodeFaceList = []
        classNames = []
        flag_GenerateStudentName = False
        if input_page_data["database_file"] is not None:
            with zipfile.ZipFile(st.session_state.uploadFile_path, 'r') as zip_ref:
                zip_ref.extractall(st.session_state.session_storage)
            unzipDatabase_Path = osp.join(st.session_state.session_storage, osp.splitext(osp.basename(st.session_state.uploadFile_path))[0]) 
            #unzipDatabase_Path = glob.glob(osp.join(st.session_state.session_storage, "*"))[0]
            print("unzipDatabase_Path {}".format(unzipDatabase_Path))
            encodeFaceList, classNames, faces_visualization = face_recognition_.encode_face(unzipDatabase_Path, faceDetector_Config, faceDetector)
            st.session_state.imagesDatabase = dict(zip(classNames, faces_visualization))
        else:
            st.session_state.imagesDatabase = {}
        if encodeFaceList == []:
            flag_GenerateStudentName = True




        while True:
            ret, img = video.read()
            if ret == True:
                img_show = img.copy()
                if control._frame_idx % frame_process_jump == 0 :
                    flag_numFace_isChanged = False
                    # Face detection 
                    faces = faceDetector_Config.detect_faces(faceDetector, img)
                    if int(len(faces))>0 and control._firstFaceDetected == True :
                        control._firstFaceDetected = True
                    if control._current_numFace != 0 and control._current_numFace != int(len(faces)):
                        flag_numFace_isChanged = True 
                    control._current_numFace = int(len(faces))
                    #faces = face_recognition_.recognize_and_re_identify_face(encodeFaceList, classNames, img, faceDetector_Config, faceDetector, control, student_list)
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    #intialize frame:
                    currentFrame = frameData_.Frame(control._frame_idx)
                    control._count_face = 0 

                    for index_face, face in enumerate(faces):
                        control._count_face+=1
                        converted_face = face["box"] if faceDetector_Config.type=="mtcnn" else list(face)
                        [x,y,w,h] = converted_face
                        face_coord = [x,y,w,h]
                        ROI_crop =  img[y:y+h, x:x+w]
                        ROI_gray = cv2.cvtColor(ROI_crop, cv2.COLOR_BGR2GRAY)
                        emotion = None
                        attention = None
                        student = None
                        name = None

                        if flag_GenerateStudentName == True:
                            if encodeFaceList == []:
                                face_CurrentFrame = [tuple([converted_face[1],converted_face[0]+converted_face[2],converted_face[1]+converted_face[3],converted_face[0]])]
                                encodes_CurrentFrame = face_recognition.face_encodings(img, face_CurrentFrame)[0]
                                encodeFaceList.append(encodes_CurrentFrame)
                                name = "S_"+str(control._set_id_face)
                                classNames.append(name)
                                control._list_Currentnames.append(name)
                                student = student_.Student(str(control._set_id_face), str(name))
                                control._set_id_face +=1
                                student._face_point = ((2*face_coord[0]+face_coord[2])//2,(2*face_coord[1]+face_coord[3])//2)
                                student._face_region = cv2.cvtColor(ROI_crop,cv2.COLOR_BGR2RGB)
                                student._num_Detected +=1
                                student._face_coord = [x,y,w,h]        
                                student_list.append(student)
                                st.session_state.imagesDatabase.update({name:student._face_region})
                            else:
                                if flag_numFace_isChanged == True:
                                    name = face_recognition_.recognize_face(encodeFaceList, classNames, img, converted_face)
                                    if name == None:
                                        face_CurrentFrame = [tuple([converted_face[1],converted_face[0]+converted_face[2],converted_face[1]+converted_face[3],converted_face[0]])]
                                        encodes_CurrentFrame = face_recognition.face_encodings(img, face_CurrentFrame)[0]
                                        encodeFaceList.append(encodes_CurrentFrame)
                                        name = "S_"+str(control._set_id_face)
                                        classNames.append(name)
                                        control._list_Currentnames.append(name)
                                        student = student_.Student(str(control._set_id_face), str(name))
                                        control._set_id_face +=1
                                        student._face_point = ((2*face_coord[0]+face_coord[2])//2,(2*face_coord[1]+face_coord[3])//2)
                                        student._face_region = cv2.cvtColor(ROI_crop,cv2.COLOR_BGR2RGB)
                                        student._num_Detected +=1
                                        student._face_coord = [x,y,w,h]        
                                        student_list.append(student)
                                        st.session_state.imagesDatabase.update({name:student._face_region})
                                    else:
                                        # if this face was detected, try to find this face in list and update it!
                                        for iter_student in student_list:
                                            if iter_student._name == str(name):
                                                student = iter_student
                                                student._face_point = ((2*face_coord[0]+face_coord[2])//2,(2*face_coord[1]+face_coord[3])//2)
                                                student._face_region = cv2.cvtColor(ROI_crop,cv2.COLOR_BGR2RGB)
                                                student._num_Detected +=1
                                                student._face_coord = [x,y,w,h]
                                                break
                                    
                                # turn on face_re-identification
                                else:
                                    face_point = ((2*face_coord[0]+face_coord[2])//2,(2*face_coord[1]+face_coord[3])//2)
                                    student = faceIdentification_.identify_face(face_point, student_list)
                                    face_coord = [x,y,w,h]
                                    student._face_point = face_point
                                    student._face_region = cv2.cvtColor(ROI_crop,cv2.COLOR_BGR2RGB)
                                    student._num_Detected +=1
                                    student._face_coord = [x,y,w,h]

                           

                        else:
                            # If this is the first time that face was detected is False
                            if control._firstFaceDetected == False:
                                # turn on face_recognition
                                # if number of face in current frame is different from previous frame
                                if flag_numFace_isChanged == True:
                                    name = face_recognition_.recognize_face(encodeFaceList, classNames, img, converted_face)
                                    if name == None:
                                        break
                                        # control._list_Currentnames.append(name)
                                        # name = "Unknown"+str(control._set_id_face)
                                        
                                        # control._list_Currentnames.append(name)
                                        # student = student_.Student(str(control._set_id_face), str(name))
                                        # control._set_id_face +=1
                                        # student._face_point = ((2*face_coord[0]+face_coord[2])//2,(2*face_coord[1]+face_coord[3])//2)
                                        # student._face_region = cv2.cvtColor(ROI_crop,cv2.COLOR_BGR2RGB)
                                        # student._face_coord = [x,y,w,h]
                                        # student_list.append(student) 
                                        # st.session_state.imagesDatabase.update({name:student._face_region})

                                    elif name not in control._list_Currentnames:
                                        # add new student into list if this face isn't in list
                                            control._list_Currentnames.append(name)
                                            student = student_.Student(str(control._set_id_face), str(name))
                                            control._set_id_face +=1
                                            student._face_point = ((2*face_coord[0]+face_coord[2])//2,(2*face_coord[1]+face_coord[3])//2)
                                            student._face_region = cv2.cvtColor(ROI_crop,cv2.COLOR_BGR2RGB)
                                            student._num_Detected +=1
                                            student._face_coord = [x,y,w,h]        
                                            student_list.append(student)  
                                    else: 
                                        # if this face was detected, try to find this face in list and update it!
                                        for iter_student in student_list:
                                            if iter_student._name == str(name):
                                                student = iter_student
                                                student._face_point = ((2*face_coord[0]+face_coord[2])//2,(2*face_coord[1]+face_coord[3])//2)
                                                student._face_region = cv2.cvtColor(ROI_crop,cv2.COLOR_BGR2RGB)
                                                student._num_Detected +=1
                                                student._face_coord = [x,y,w,h]
                                                break
                                    
                                # turn on face_re-identification
                                else:
                                    face_point = ((2*face_coord[0]+face_coord[2])//2,(2*face_coord[1]+face_coord[3])//2)
                                    student = faceIdentification_.identify_face(face_point, student_list)
                                    face_coord = [x,y,w,h]
                                    student._face_point = face_point
                                    student._face_region = cv2.cvtColor(ROI_crop,cv2.COLOR_BGR2RGB)
                                    student._num_Detected +=1
                                    student._face_coord = [x,y,w,h]
                                    
                            else:
                                name = face_recognition_.recognize_face(encodeFaceList, classNames, img, converted_face)
                                if name == None:
                                    break
                                control._list_Currentnames.append(name)
                                student = student_.Student(str(control._set_id_face), str(name))
                                control._set_id_face +=1
                                student._face_point = ((2*face_coord[0]+face_coord[2])//2,(2*face_coord[1]+face_coord[3])//2)
                                student._face_region = cv2.cvtColor(ROI_crop,cv2.COLOR_BGR2RGB)
                                student._num_Detected +=1
                                student._face_coord = [x,y,w,h]
                                student_list.append(student)  

                        if (name == None and flag_numFace_isChanged == True) or (name == None and control._firstFaceDetected == True):
                            continue

                        # ANALYSE
                        if config.turnOn_emotion_detection:
                            #Emotion detection
                            emotion = emotionDetector_Config.detect_emotions(emotionDetector, ROI_gray)
                            student._emotion[emotion]+=1

                        if config.turnOn_eyeGaze_estimation:
                            # Landmasks detection
                            face_data = face_landMarkDetection_Config.detect_landMarks(face_LandMarkDetector, converted_face, gray_img)
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
                                attention = "focused"
                                student._status_eyeGaze_count = 0
                                #print("focus - {}".format(attention))
                            else:
                                if student._status_eyeGaze_count <=3:
                                    attention = "focused"
                                    student._status_eyeGaze_count +=1
                                else:
                                    attention = "distracted"
                                #print("distracted - {}".format(attention))
                            student._attention[attention]+=1
                        
                        #level_engagement, _ = random.choice(list(student._engagement_level.items()))
                        #student._emotion[random.choice(list(student._emotion.items()))[0]]+=1
                        #student._attention[random.choice(list(student._attention.items()))[0]]+=1
                        #student._engagement_level[random.choice(list(student._engagement_level.items()))[0]]+=1
                        
                        
                        # CALCULATE ENGAGEMENT SCORE
                        engagement_level = engagementDetection_.detect_engagement(emotion, attention)
                        student._engagement_level[engagement_level]+=1
                        #st.write(engagement_level)

                        #save emotion and engagement level for every frame
                        currentFrame._emotion[emotion] +=1
                        currentFrame._attention[attention] +=1
                        currentFrame._engagement_level_mapping_method[engagement_level] +=1

                        # save the results
                        #student._id = "21R"+"%03d" %int(student._name)
                        student._id = str(student._name)
                        student.analytic_student_infos(emotion=emotion, attention=attention, engagement_level=engagement_level)
                        database_conn = connect_db.create_connection(osp.join(project_path, st.session_state.config_webservice.database_path))
                        control._ID_analytic_row+=1
                        #face_id = "%06d_%06d"%(control._frame_idx, int(student._name))
                        face_id = "%06d_%s"%(control._frame_idx, str(student._name))
                        storage.save_student_El_Mapping_Method(database_conn, control._ID_analytic_row, control._frame_idx, face_id, student)
                        database_conn.close()
                        # DRAW ON FRAME
                        if config.drawOnFrame is True:
                            if config.turnOn_emotion_detection:
                                draw_EmotionLabel(img_show,x, y, w, h, emotion, emotionDetector_Config.emotion_color_dict[emotion])

                            #if config.turnOn_eyeGaze_estimation:
                            # Draw EyeGaze line
                                #if attention == "focused":
                                #    draw_AttentionLabel(img_show, x, y, w, h, "looking at screen", (0, 255, 0))
                                #else:
                                #    draw_AttentionLabel(img_show, x, y, w, h, "looking away from the screen", (0, 0, 255))
                                #draw_EyeGageLine(img_show, eye_centers_ord, eye_view_ord, (0, 255, 255))
                            if config.turnOn_face_identification:
                            # Draw student_name label
                                #draw_NameLabel(img_show, x, y, w, h, "21R"+"%03d" %int(student._name), (255,255,255))
                                draw_NameLabel(img_show, x, y, w, h, str(student._name), (255,255,255))
                            

                            if attention == "focused" :
                                #draw_BoundingBox(img_show, x, y, w, h, (0,255,255))
                                draw_BoundingBox(img_show, x, y, w, h, engagement_level_color[engagement_level])
                            else:
                                draw_BoundingBox(img_show, x, y, w, h, (0,0,255))
                            st.session_state.student_list = student_list

                    if int(len(faces))>0:
                        control._firstFaceDetected = False
                    control._ID_analytic_frameData_row+=1
                    database_conn = connect_db.create_connection(osp.join(project_path, st.session_state.config_webservice.database_path))
                    if config.method_mapping == "concentration_index":
                        storage.save_class_el_summary_KES_method(database_conn, control._ID_analytic_frameData_row, currentFrame)
                    elif config.method_mapping == "traditional_mapping":
                        storage.save_class_el_summary_Mapping_method(database_conn, control._ID_analytic_frameData_row, currentFrame)
                    database_conn.close()
                    # calculate the running time
                    currentTime = time.time()
                    delay2Frame = float(currentTime-previousTime)
                    previousTime = currentTime
                    #print("{}: {}".format(control._frame_idx, round(delay2Frame,2)))
                    fps = int(1/delay2Frame)
                    fps_str = str(fps)+"|"+str(int(inputSource.fps))


                    
                    # save the resuls
                    st.session_state.img_show = img_show
                    if st.session_state.config_webservice.write2File is True:
                        hour, minute_res, second_res = calculate_time_duration(int(inputSource.fps), int(control._frame_idx))
                        timeline_point = str(hour)+":"+str(minute_res)+":"+str(second_res)
                        print(timeline_point)
                        cv2.putText(st.session_state.img_show,str(timeline_point),(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)
                        st.session_state.out_video.write(st.session_state.img_show)
                if st.session_state.isRunning == False:
                    break
                # analyze results
                time_second_processed = int(int(control._frame_idx // inputSource.fps)//time_per_second)
                if (time_second_processed % time_to_visualize == 0) and ((time_second_processed // time_to_visualize) != index_time_analyze):
                    #print("time_second_processed {}".format(time_second_processed))
                    #print("control._frame_idx {}".format(control._frame_idx))
                    #print("time_to_visualize {}".format(time_to_visualize))
                    print("-"*20)

                    index_time_analyze = (time_second_processed // time_to_visualize)
                    hour, minute_res, second_res = calculate_time_duration(int(inputSource.fps), int(control._frame_idx))
                    timeline_point = str(hour)+":"+str(minute_res)+":"+str(second_res)
                    timeline_.append(timeline_point)


                    # visualize engagement analytic results#
                    # -----------
                    engagement_bar_chart_value = analyze_session_engagement_(student_list)
                    data_Engagement = data_Engagement.append(pd.DataFrame([engagement_bar_chart_value], columns=['strong engagement','high engagement','medium engagement','low engagement','disengagement']), ignore_index=True)
                    st.session_state.data_Engagement = data_Engagement
                    st.session_state.timeline_data_Engagement = timeline_
                    # settings
                    fig = px.bar(data_Engagement, 
                                template = "none",
                                x = timeline_,
                                y = [c for c in data_Engagement.columns],
                                color_discrete_map={'strong engagement': '#C7DBFF', 'medium engagement': '#6FA4FF', 'high engagement': '#8DB6FF', 'low engagement': '#5291FF', 'disengagement': '#bf360c'},
                                labels={'value':'Number of student', 'x':'Timeline', 'variable':'Engagement level'},
                                height=380
                                )
                    fig.update_layout(barmode='stack',font=dict(size=16), margin=dict(t=10),plot_bgcolor='rgba(0,0,0,0)',hovermode="x")
                    # visualize figure
                    engagement_chart_placeholder.plotly_chart(fig, use_container_width=True)

                    # visualize emotion analytic results
                    # -----------
                    data_EmotionScore = analyze_emotion_(student_list, total_analytic_emotion)
                    st.session_state.data_EmotionScore = data_EmotionScore

                    # settings
                    data = go.Pie(labels = data_EmotionLabel, values = data_EmotionScore)
                    layout = go.Layout(font=dict(size=14),autosize=False,width=400,height=400, 
                            xaxis= go.layout.XAxis(linecolor = 'black', linewidth = 1, mirror = True),
                            yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 3, mirror = True),
                            margin=go.layout.Margin(l=50, r=50, b=100, t=10,pad = 4))
                    fig = go.Figure(data = [data] ,layout=layout)
                
                    colors = ['#EE6666', '#5470C6','#73C0DE','#FAC858','#3BA272','#91CC75','#FC8452']
                    fig.update_traces(hoverinfo='label+value', textinfo='label+percent',marker=dict(colors=colors))
                    # visualize figure
                    emotion_chart_placeholder.plotly_chart(fig, use_container_width=False)

                    # visualize concentrantion analytic results
                    # -----------
                    concentration_chart_value = analyze_session_concentration_(student_list)
                    data_ConcentrationData = data_ConcentrationData.append(pd.DataFrame([concentration_chart_value], columns=['Focused', "Distracted"]), ignore_index=True)
                    st.session_state.data_ConcentrationData = data_ConcentrationData
                    # settings
                    fig = px.area(data_ConcentrationData, 
                                template = "none",
                                x = timeline_,
                                y = [c for c in data_ConcentrationData.columns], 
                                height=380,
                                labels={'value':'Number of student', 'x':'Timeline', 'variable':'Concentration level'},
                                )
                    fig.update_layout(barmode='stack',font=dict(size=16), margin=dict(t=10),plot_bgcolor='rgba(0,0,0,0)')
                    # visualize figure
                    concentration_chart_placeholder.plotly_chart(fig, use_container_width=True)

                    # settings
                    #fig = px.line(data_ConcentrationData, template = "none",x = data_ConcentrationData.index, y = [c for c in data_ConcentrationData.columns], 
                    #            labels={'value':'Number of student', 'index':'Timeline', 'variable':'Concentration type'},height=340)
                    #fig.update_layout(autosize=True, margin=dict(t=10, l=30),plot_bgcolor='rgba(0,0,0,0)')
                    #fig.update_traces(mode='markers+lines')
                    # visualize figure
                    #concentration_chart_placeholder.plotly_chart(fig, use_container_width=True)
                    

                    if input_page_data["show_frame"] is False:
                        # visualize top engagement students
                        image_template = Image.open(osp.join(project_path,'media/student.jpg'))
                        list_top_engement_student_dict = analyze_session_top_engagement_(student_list)
                        img_top_engagement_student = []
                        cap_top_engagement_student = []
                        if list_top_engement_student_dict == {}:
                            cap_top_engagement_student.append("N/A")
                            img_top_engagement_student.append(image_template)
                        else:
                            for name_student, face_ROI in list_top_engement_student_dict.items():
                                cap_top_engagement_student.append(str(name_student))
                                img_top_engagement_student.append(face_ROI)

                        list_top_disengement_student_dict = analyze_session_top_disengagement_(student_list)
                        img_top_disengagement_student = []
                        cap_top_disengagement_student = []
                        if list_top_disengement_student_dict == {}:
                            cap_top_disengagement_student.append("N/A")
                            img_top_disengagement_student.append(image_template)
                        else:
                            for name_student, face_ROI in list_top_disengement_student_dict.items():
                                cap_top_disengagement_student.append(str(name_student))
                                img_top_disengagement_student.append(face_ROI)

                        if img_top_engagement_student ==[]:
                            img_top_engagement_student = [image_template]

                        if img_top_disengagement_student ==[]:
                            img_top_disengagement_student = [image_template]
                        
                        img_top_engagement_student_plholder.image(img_top_engagement_student, width=90, caption = cap_top_engagement_student)
                        img_top_disengagement_student_plholder.image(img_top_disengagement_student, width=90, caption = cap_top_disengagement_student)
                    else:
                        converted_img_show = cv2.cvtColor(st.session_state.img_show, cv2.COLOR_BGR2RGB)
                        show_frame_plholder.image(converted_img_show)

                    reset_analytic_results_all(student_list)
                    # Frame Post-Process
                    #if st.session_state.config_webservice.write2File is True:
                    #    st.session_state.out_video.write(img_show)
                    #    st.session_state.img_show = img_show
                #if control._frame_idx >=50000 or control._frame_idx > (int(inputSource.length)-int(inputSource.fps)):
                #    break
                control._frame_idx +=1

                    #st.session_state.img_show = img_show
                #process_value = float(int(control._frame_idx)*100/int(inputSource.length))
                #progress_bar.progress(process_value)
                #status_running_placeholder.write("Processing... {}%".format(round(process_value,2)))
            else: 
                st.session_state.video.release()
                if st.session_state.config_webservice.write2File is True:
                    st.session_state.out_video.release()
                
                st.session_state.status_running_placeholder.success("Finished processing! Click STOP button to generate the report")
                break

    elif input_page_data["start_button"] is True and st.session_state.isStart is True:
        st.session_state.video.release()
        if st.session_state.config_webservice.write2File is True:
            st.session_state.out.release()
        st.write("Finished!")

def Main():
    
    #debug
    #st.session_state.login_gate = True
    #st.session_state.username = "admin"

    # SET CONFIGURES
    hide_menu_button(True)  # Hide menu button
    #hide_stop_button(True)  # Hide stop button
    if 'config_webservice' not in st.session_state:
        config_webservice = initialize_Config_Webservice()
        st.session_state.config_webservice = config_webservice
    
    # LOGIN PAGE : 
    if "login_gate" not in st.session_state or st.session_state.login_gate is False:
        

        # initilize default account if not exist 
        if "username" not in st.session_state:
            st.session_state.username = "user2"
            st.session_state.userFullname = "Admin"
        if "password" not in st.session_state:
            st.session_state.password = "user2"

        if "isSignUp" not in st.session_state:
            LogIn_page()
        elif st.session_state.isSignUp is True: 
            SignUp_page()
            
    elif st.session_state.login_gate is True:
        if "isFirstTime" not in st.session_state:
            st.session_state.isFirstTime = True
        if "isRunning" not in st.session_state:
            st.session_state.isRunning = False
        if st.session_state.username == "admin": 
            input_page_data = Input_page_for_admin()
            if st.session_state.isRunning is True:
                Main_page(input_page_data)
            elif st.session_state.isFirstTime is False:
                Summary_page(input_page_data)
        else:
            input_page_data = Input_page_for_user()
            if st.session_state.isRunning is True:
                Main_page(input_page_data)
            elif st.session_state.isFirstTime is False:
                Summary_page(input_page_data)

    # INPUT PAGE
    #input_page_data = Input_page()
    
    # MAIN PAGE
    #Main_page(input_page_data)

    # SUMMAR

if __name__=='__main__':
    Main()

