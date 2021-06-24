import os
import sys
import cv2
import numpy as np
import os.path as osp
import tensorflow as tf

class FaceDetection:
    def __init__(self):
        # basic infos
        self.type = None # haarcascade|mtcnn
        self.weight_path = None
        self.path = None

    def process(self, config):
        # Check input_video is recored video or streaming video
        if config.type_faceDetection == "haarcascade_faceDetection":
            self.type = "haarcascade_faceDetection"
            self.path = osp.join(config.algorithm_path,self.type)
            self.weight_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        elif config.type_faceDetection == "mtcnn":
            self.type = "mtcnn"
            self.path = osp.join(config.algorithm_path,self.type)
        else:
            print("[Error] The type of faceDetection is unknown!")
        # process
    
    def intialize(self):
        sys.path.append(self.path)
        if self.type == "haarcascade_faceDetection":
            return cv2.CascadeClassifier(self.weight_path)
        elif self.type == "mtcnn":
            from mtcnn import MTCNN
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            return MTCNN()
    
    def detect_faces(self, faceDetector, frame):
        if self.type == "haarcascade_faceDetection":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceDetector.detectMultiScale(frame, 1.7, 5)
            faces = list(faces)
        elif self.type == "mtcnn":
            faces = faceDetector.detect_faces(frame)
        # convert to standard  
        return faces



