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
# ------------
import Api_tools.IO_file as IO_file # import api_tools
from tools import *                     # import tools
# -Import Objects-
from Objects import config_
from Objects import config_webservice_
from Objects import inputSource_
from Objects import faceDetection_
from Objects import emotionDetection_
from Objects import face_landMarkDetection_
from Objects import control_

# INITIALIZE
def initialize_Config():
    # Read config
    print("Read config file ...",end=" ")
    parsed_config = IO_file.read_config(osp.join(os.path.dirname(os.path.abspath(__file__)),"config.yml"))
    config = config_.Config()
    config = config.import_config(parsed_config)
    print("Ok")
    return config
 
def initialize_Config_Webservice():
    # Read config
    print("Read config webservice file ...",end=" ")
    #parsed_config = IO_file.read_config("/home/hont/Engagement_Detection/config_webservice.yml")
    parsed_config = IO_file.read_config(osp.join(os.path.dirname(os.path.abspath(__file__)),"config_webservice.yml"))
    config = config_webservice_.Config_Webservice()
    config = config.import_config(parsed_config)
    print("Ok")
    return config

def initialize_inputSource(config):
    # Process source_video
    print("Process input_source ...",end=" ")
    inputSource = inputSource_.InputSource()
    inputSource.process(config)
    print("Ok")
    return inputSource

def initialize_Control(config):
    print("Process control ...",end=" ")
    control = control_.Control()
    control.process(config)
    print("Ok")
    return control

# INITIALIZE_TECHNIQUES
def initialize_faceDetector(config):
    print("intialize faceDetector ...",end=" ")
    faceDetector_Config = faceDetection_.FaceDetection()
    faceDetector_Config.process(config)
    faceDetector = faceDetector_Config.intialize()
    print("OK")
    return faceDetector_Config, faceDetector

def initialize_emotionDetector(config):
    print("intialize emotionDetector ...",end=" ")
    emotionDetector_Config = emotionDetection_.EmotionDetection()
    emotionDetector_Config.process(config)
    emotionDetector = emotionDetector_Config.intialize()
    print("OK")
    return emotionDetector_Config, emotionDetector

def initialize_face_landMarkDetector(config):
    print("intialize face_landMarkDetector ...",end=" ")
    face_landMarkDetection_Config = face_landMarkDetection_.Face_LandMarkDetection()
    face_landMarkDetection_Config.process(config)
    face_LandMarkDetector = face_landMarkDetection_Config.intialize()
    print("OK")
    return face_landMarkDetection_Config, face_LandMarkDetector


# INITIALIZE_INFORMATION
def initialize_random_student(faces):
    pass