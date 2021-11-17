import streamlit as st
from datetime import datetime
import os
import time

def switch_state_processing(Output_video_path, coverted_output_video_path):
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
    input_source = input_page_container.text_input("Enter the URL of video or streaming", "/home/hont/input_video/input0.mp4")
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
    database_file = input_page_container.file_uploader("Database for face recognition (Optional)",type=['CSV'])
    if database_file is not None:
        file_details = {"FileName":database_file.name,"FileType":database_file.type,"FileSize":database_file.size}
        input_page_container.write(file_details)

    # select which method to mapping
    mappingMethod_type =  input_page_container.selectbox("Select method to map the features", ( 'Pseudo mapping', 'KES mapping',))

    # check settings
    send_email_checkbox = input_page_container.checkbox('Send a Summary Report to my email', True) # visualize image
    export_video_checkbox = input_page_container.checkbox('Write to ouput video ') # export to output video
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
    database_file = input_page_container.file_uploader("Database for face recognition (Optional)",type=['CSV'])
    if database_file is not None:
        file_details = {"FileName":database_file.name,"FileType":database_file.type,"FileSize":database_file.size}
        input_page_container.write(file_details)

    # select which method to mapping
    mappingMethod_type =  input_page_container.selectbox("Select method to map the features", ( 'Pseudo mapping', 'KES mapping',))

    # check settings
    send_email_checkbox = input_page_container.checkbox('Send a Summary Report to my email', True) # visualize image
    export_video_checkbox = input_page_container.checkbox('Write to ouput video ') # export to output video
    #debug_mode = input_page_container.checkbox('Debug mode (process 1 time/s)') # debug mode
    debug_mode = False
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
                        "email address":"nguyentanhoit@gmail.com",
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
