import streamlit as st
import base64
import time
# -Import Database API-
from Database import connect_db
from Database import api_db

def processing_time_slider(max_fps):
    processing_time  = st.sidebar.slider('Number of Frames to process in one second', 1, 3, max_fps)
    data = {'processing_time':processing_time}
    return data

def calculate_time_duration(fps, numFPS):
    second = int(numFPS/int(fps))
    minute = int(second/60)
    second_res = int(second%60)
    hour = int(minute/60)
    minute_res = int(minute%60)
    return hour, minute_res, second_res

def calculate_percent_emotion_type(total_emotion):
    sum_ = sum(total_emotion.values())
    percent_arr = []
    if sum_==0:
        for idx, emotion in enumerate(total_emotion):
            percent_arr.append(0)
    else:
        percent_arr = []
        for idx, emotion in enumerate(total_emotion):
            if idx != len(total_emotion)-1:
                percent_value = int((total_emotion[emotion] * 100)/sum_)
                percent_arr.append(percent_value)
            else: 
                percent_value = 100 - sum(percent_arr[:])
                percent_arr.append(percent_value)
    return percent_arr

def analyze_emotion_(student_list, total_analytic_emotion):
    #total_session_emotion = {"Angry": 0, "Disgusted": 0, "Fearful" :0, "Happy":0, "Neutral":0, "Sad":0, "Surprised":0}
    for idx, student_ in enumerate(student_list):
        total_analytic_emotion[max(student_._emotion, key=student_._emotion.get)]+=1
    return calculate_percent_emotion_type(total_analytic_emotion)

def analyze_session_engagement_(student_list):
    total_session_engagement = {"strong engagement":0, "medium engagement":0, "high engagement":0, "low engagement": 0, "disengagement":0}
    for idx, student_ in enumerate(student_list):
        max_level = max(student_._engagement_level, key=student_._engagement_level.get)
        if student_._engagement_level[max_level]!=0:
            total_session_engagement[max_level]+=1
    return list(total_session_engagement.values())

def analyze_session_concentration_(student_list):
    total_session_concentration = {"focused":0, "distracted":0}
    total_session_engagement = {"strong engagement":0, "medium engagement":0, "high engagement":0, "low engagement": 0, "disengagement":0}
    for idx, student_ in enumerate(student_list):
        max_level = max(student_._engagement_level, key=student_._engagement_level.get)
        if student_._engagement_level[max_level]!=0:
            if max_level!="disengagement":
                total_session_concentration["focused"]+=1
            else:
                total_session_concentration["distracted"]+=1
    return list(total_session_concentration.values())

    """
    total_session_concentration = {"focused":0, "distracted":0}
    for idx, student_ in enumerate(student_list):
        max_level = max(student_._attention, key=student_._attention.get)
        if student_._attention[max_level]!=0:
            total_session_concentration[max_level]+=1
    """
    return list(total_session_concentration.values())

def reset_analytic_results_all(student_list):
    for idx, student in enumerate(student_list):
        student._reset_analytic_results()

def analyze_session_top_engagement_(student_list):
    list_top_student_dict = {}
    for idx, student_ in enumerate(student_list):
        max_level = max(student_._engagement_level, key=student_._engagement_level.get)
        #if engagement_level== "strong engagement" or engagement_level== "high engagement":
        if max_level== "strong engagement" and student_._engagement_level[max_level]!=0:
            list_top_student_dict.update({student_._name:student_._face_region})
            if len(list_top_student_dict)>=5:
                return list_top_student_dict
    return list_top_student_dict

def analyze_session_top_disengagement_(student_list):
    list_top_student_dict = {}
    for idx, student_ in enumerate(student_list):
        max_level = max(student_._engagement_level, key=student_._engagement_level.get)
        if max_level == "disengagement" and student_._engagement_level[max_level]!=0:
            list_top_student_dict.update({student_._name:student_._face_region})
            if len(list_top_student_dict)>=5:
                return list_top_student_dict
    
    return list_top_student_dict

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

def hide_menu_button(flag):
    if flag == True:
        st.markdown(""" <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style> """, unsafe_allow_html=True)

def title_bar():
    LOGO_IMAGE = '/home/hont/Engagement_Detection/media/logo.png'
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
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            <p class="logo-text">ENGAGEMENT DETECTION SYSTEM</p>
        </div>
        <hr class="line">
        """,
        unsafe_allow_html=True
    )

# database api 
def reset_tables(config, config_webservice):
    database_conn = connect_db.create_connection(config_webservice.database_path)
    if config.method_mapping == "concentration_index":
        api_db.reset_table(database_conn, "STUDENT_EL_KES_METHOD")
        api_db.reset_table(database_conn, "CLASS_EL_SUMMARY_KES_METHOD")
    elif config.method_mapping == "traditional_mapping":
        api_db.reset_table(database_conn, "STUDENT_EL_MAPPING_METHOD")
        api_db.reset_table(database_conn, "CLASS_EL_SUMMARY_MAPPING_METHOD")
    database_conn.close()
