import streamlit as st



# Select the number of frame to be process in one second
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
        total_session_engagement[max(student_._engagement_level, key=student_._engagement_level.get)]+=1
    return list(total_session_engagement.values())

def reset_analytic_results_all(student_list):
    for idx, student in enumerate(student_list):
        student._reset_analytic_results()

def analyze_session_top_engagement_(student_list):
    list_top_student_dict = {}
    for idx, student_ in enumerate(student_list):
        engagement_level = max(student_._engagement_level, key=student_._engagement_level.get)
        if engagement_level== "strong engagement" or engagement_level== "high engagement":
            list_top_student_dict.update({student_._name:student_._face_region})
            if len(list_top_student_dict)>=5:
                return list_top_student_dict
    return list_top_student_dict

def analyze_session_top_disengagement_(student_list):
    list_top_student_dict = {}
    for idx, student_ in enumerate(student_list):
        engagement_level = max(student_._engagement_level, key=student_._engagement_level.get)
        if engagement_level == "low engagement" or engagement_level == "disengagement" :
            list_top_student_dict.update({student_._name:student_._face_region})
            if len(list_top_student_dict)>=5:
                return list_top_student_dict
    return list_top_student_dict

