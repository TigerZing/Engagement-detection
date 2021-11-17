from Api_tools import *
import cv2





def modify_config(config, source_video, techniques):
    config.source_video = source_video
    config.type_faceDetection = techniques["Face_Detection"]
    config.type_landMaskDetection = techniques["LandMark_Detection"]
    config.type_eyeGazeEstimation = techniques["EyeGaze_Detection"]
    config.type_emotionDetection = techniques["Emotion_Detection"]
    return config


# --------------------------------------
# Draw bouding box
def draw_BoundingBox(img,x, y, w, h, color, line=3):
    cv2.rectangle(img,(x,y),(x+w,y+h),color,line)

# Draw emotion-label 
def draw_EmotionLabel(img, x, y, w, h, emotion, color):
    if y > 50:
        cv2.rectangle(img,(x, y-45),(x+len(emotion)*20 -len(emotion)*2 ,y-15),(0,255,255),-1)
        cv2.putText(img,  emotion, (x, y-20), cv2.FONT_HERSHEY_DUPLEX , 1, color ,2, cv2.LINE_4)
    else: 
        cv2.rectangle(img,(x, y+h+15),(x+len(emotion)*20 -len(emotion)*2 ,y+h+45),(0,255,255),-1)
        cv2.putText(img,  emotion, (x, y+h+40), cv2.FONT_HERSHEY_DUPLEX , 1, color ,2, cv2.LINE_4)

# Draw attention-label 
def draw_AttentionLabel(img, x, y, w, h, status, color):
    if y > 50:
        cv2.rectangle(img,(x, y-45),(x+len(status)*20 -len(status)*2 ,y-15),(255,255,255),-1)
        cv2.putText(img,  status, (x, y-20), cv2.FONT_HERSHEY_DUPLEX , 1, color ,2, cv2.LINE_4)
    else: 
        cv2.rectangle(img,(x, y+h+15),(x+len(status)*20 -len(status)*2 ,y+h+45),(255,255,255),-1)
        cv2.putText(img,  status, (x, y+h+40), cv2.FONT_HERSHEY_DUPLEX , 1, color ,2, cv2.LINE_4)

# Draw name-label 
def draw_NameLabel(img, x, y, w, h, student_name, color):
    if y > 50:
        cv2.rectangle(img,(x, y+h+5),(x+5+len(student_name)*20 -len(student_name)*2 ,y+h+35),color,-1)
        cv2.putText(img,  student_name, (x, y+h+31), cv2.FONT_HERSHEY_DUPLEX , 1,(0,0,0) ,2, cv2.LINE_4)
    else: 
        cv2.rectangle(img,(x, y+h+50),(x+5+len(student_name)*20 -len(student_name)*2 ,y+h+85),color,-1)
        cv2.putText(img,  student_name, (x, y+h+82), cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,0) ,2, cv2.LINE_4)

# Draw EyeGage Line
def draw_EyeGageLine(img, eye_centers_ord, eye_view_ord, color):
    cv2.line(img, eye_centers_ord, eye_view_ord, color, 2, -1) # detect landmasks

# Draw running time
def draw_runningTime(img, fps, width, height):
    #cv2.rectangle(img,(x, y-45),(x+len(emotion)*20 -len(emotion)*2 ,y-15),(0,255,255),-1)
    cv2.putText(img, fps, (int(width) - 110, 30), cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,255) ,2, cv2.LINE_4)
