import os
import sys
import cv2
import numpy as np
import os.path as osp
import yaml
import io


class Config:
    def __init__(self):
        # basic configs
        self.project_path = None
        self.algorithm_path = None
        self.database_path = None
        self.time_process = 1 # Every time_process seconds pickup 1 frame to process
        self.source_video = None
        # kind of algorithms configs
        self.type_faceDetection = None
        self.type_emotionDetection = None
        self.type_landMaskDetection = None
        self.type_eyeGazeEstimation = None
        # algorithm settings configs
        self.turnOn_face_identification = False
        self.turnOn_emotion_detection = False
        # database configs
        self.turnOn_storageDB = False
        self.turnOn_resetDB = False
        # processing settings
        self.write2File: False
        self.outputFile_path = None
        self.showFrame = False
        self.drawOnFrame = True 
        self.drawRunningTime = False

    # import info to object [Config]
    def import_config(self, parsed_config):
        # basic configs
        self.project_path = parsed_config["basic_config"]["project_path"]
        self.algorithm_path = osp.join(self.project_path,"Algorithms")
        self.database_path = osp.join(self.project_path,"Database")
        self.time_process = parsed_config["basic_config"]["time_process"]
        self.source_video = parsed_config["basic_config"]["source_video"]
        # kind of algorithms configs
        self.type_faceDetection = parsed_config["type_config"]["type_faceDetection"]
        self.type_emotionDetection = parsed_config["type_config"]["type_emotionDetection"]
        self.type_landMaskDetection = parsed_config["type_config"]["type_landMaskDetection"]
        self.type_eyeGazeEstimation = parsed_config["type_config"]["type_eyeGazeEstimation"]
        # algorithm settings configs
        self.turnOn_face_identification = parsed_config["algorthms_settings_config"]["turnOn_face_identification"]
        self.turnOn_emotion_detection = parsed_config["algorthms_settings_config"]["turnOn_emotion_detection"]
        self.turnOn_eyeGaze_estimation = parsed_config["algorthms_settings_config"]["turnOn_eyeGaze_estimation"]
        # database configs
        self.turnOn_storageDB = parsed_config["database_config"]["turnOn_storageDB"]
        self.turnOn_resetDB = parsed_config["database_config"]["turnOn_resetDB"]
        # processing settings
        self.write2File = parsed_config["processing_settings"]["write2File"]
        self.outputFile_path = parsed_config["processing_settings"]["outputFile_path"]
        self.showFrame = parsed_config["processing_settings"]["showFrame"]
        self.drawOnFrame = parsed_config["processing_settings"]["drawOnFrame"]
        self.drawRunningTime = parsed_config["processing_settings"]["drawRunningTime"]

        
        return self