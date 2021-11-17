import os
import sys
import cv2
import numpy as np
import os.path as osp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
EMOTIONS_TRANSLATE = {"angry":"Angry","disgust":"Disgusted","scared":"Fearful", "happy":"Happy", "sad":"Sad", "surprised":"Surprised","neutral":"Neutral"}
class EmotionDetection:
    def __init__(self):
        # basic infos
        self.type = None # haarcascade|mtcnn
        self.weight_path = None
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                    4: "Neutral", 5: "Sad", 6: "Surprised"}
        self.emotion_color_dict = {"Angry":(0,0,255),"Disgusted":(0,0,255),
                            "Fearful":(255,0,255),"Happy":(51,153,0),
                            "Neutral":(0,0,0),"Sad":(255,255,0),
                            "Surprised":(204,102,0)}
        self.path = None

    def process(self, config):
        # Check input_video is recored video or streaming video
        if config.type_emotionDetection == "haarcascade_emotionDetection":
            self.type = "haarcascade_emotionDetection"
            self.path = osp.join(config.algorithm_path,self.type)
            self.weight_path = osp.join(self.path,'model.h5')
        elif config.type_emotionDetection =="mini_xception_emotionDetection":
            self.type = "mini_xception_emotionDetection"
            self.path = osp.join(config.algorithm_path,self.type)
            self.weight_path = osp.join(self.path,'_mini_XCEPTION.36-0.61.hdf5')
        else:
            print("[Error] The type of faceDetection is unknown!")
        # process
    
    def intialize(self):
        sys.path.append(self.path)
        if self.type == "haarcascade_emotionDetection":
                model = Sequential()
                model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
                model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

                model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

                model.add(Flatten())
                model.add(Dense(1024, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(7, activation='softmax'))
                model.load_weights((self.weight_path))
                return model
        elif self.type == "mini_xception_emotionDetection":
            emotion_classifier = load_model(self.weight_path, compile=False)
            return emotion_classifier
    
    def detect_emotions(self, emotionDetector, roi_gray):
        maxindex = None
        if self.type == "haarcascade_emotionDetection":
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0) # resize gray face to feed into network
            prediction = emotionDetector.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            return self.emotion_dict[maxindex]
        if self.type == "mini_xception_emotionDetection":
            roi = cv2.resize(roi_gray, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotionDetector.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS_TRANSLATE[EMOTIONS[preds.argmax()]]
            return label



