import streamlit as st
import streamlit.components.v1 as components
from PIL import Image




def layout_FaceDetection(title="Face detection"):
    # title page
    st.header(title)

    #set layout
    col_chart, col_info = st.beta_columns([3,2])
    with col_chart.beta_container():
        chart_plholder = st.empty()
        showTime_plholder = st.empty()

    with col_info.beta_container():
        status_plholder = st.empty()
        info_processing_plholder = st.empty()
        fps_plholder = st.empty()

    with st.beta_container():
        info_algorithm_plholder = st.empty()
    
    return chart_plholder, showTime_plholder, status_plholder, info_processing_plholder, fps_plholder, info_algorithm_plholder

if __name__=="__main__":
    layout_FaceDetection()