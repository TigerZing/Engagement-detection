# Import libraries
import sqlite3 as lite
import os
import os.path as osp
import sys
from sqlite3 import Error
from . import connect_db
from . import api_db

# Storage_emotionalStudent_
def save_emotionalStudent(conn, emotion):
    #conn = None
    #conn = connect_db.create_connection(DATABASE_path)
    with conn:
        cur = conn.cursor()
        api_db.insert_table(cur, "EMOTION", ["NULL", str(emotion.id_student), str(emotion.id_class), str(emotion.index_frame),\
              str(emotion.emotion), str(emotion.face_box), str(emotion.face_id), str(emotion.face_point)])

def save__emotionTimelineInfo(config, listEmotionTimeline):
    conn = None
    conn = connect_db.create_connection(config.database_path)
    with conn:
        cur = conn.cursor()
        for emotionTimeline in listEmotionTimeline:
            api_db.insert_table(cur, "CLASS_EMOTION", ["NULL", (emotionTimeline.id_class), emotionTimeline.index_frame, emotionTimeline.emotions['Angry'],emotionTimeline.emotions['Disgusted'],emotionTimeline.emotions['Fearful'],emotionTimeline.emotions['Happy'],emotionTimeline.emotions['Neutral'],emotionTimeline.emotions['Sad'],emotionTimeline.emotions['Surprised']])
    conn.close()

def save_student_El_KES_Method(cur, id, indexFrame, face_id, currentStudent):
    face_id = str(currentStudent._class_id) + "_" + "%06d"%indexFrame + "_" + "%06d"%face_id
    indexFrame = int(indexFrame)
    api_db.insert_table(cur,"STUDENT_EL_KES_METHOD",[id, str(currentStudent._id), str(currentStudent._class_id), str(indexFrame),\
                                                    str(currentStudent._face_region), str(face_id), str(currentStudent._emotion), \
                                                    str(currentStudent._emotion_weight), str(currentStudent._eye_gaze_weight),\
                                                    str("%0.2f"%currentStudent._concentration_index), str(currentStudent._engagement_level)])
                            

def save_class_el_summary_KES_method(cur, id, currrentFrameData, class_id = None):
    api_db.insert_table(cur,"CLASS_EL_SUMMARY_KES_METHOD",[id,str(class_id),str(currrentFrameData._frame_id),str(currrentFrameData._emotion["Angry"]),\
                                                                            str(currrentFrameData._emotion["Disgusted"]),str(currrentFrameData._emotion["Fearful"]),str(currrentFrameData._emotion["Happy"]),\
                                                                            str(currrentFrameData._emotion["Neutral"]),str(currrentFrameData._emotion["Sad"]),str(currrentFrameData._emotion["Surprised"]),\
                                                                            str(currrentFrameData._engagement_level_KES_method["disengagement"]),str(currrentFrameData._engagement_level_KES_method["engagement"]),\
                                                                            str(currrentFrameData._engagement_level_KES_method["high engagement"])])

def save_student_El_Mapping_Method(conn, id, indexFrame, face_id, currentStudent):
    api_db.add_row_v2(conn,"STUDENT_EL_MAPPING_METHOD",[id, str(currentStudent._id), str(currentStudent._class_id), str(indexFrame),\
                                                    str(currentStudent._face_coord), str(face_id), str(currentStudent._emotion_momment),\
                                                    str(currentStudent._attention_momment), str(currentStudent._engagement_level_momment)])
                            

def save_class_el_summary_Mapping_method(conn, id, currrentFrameData,class_id = None):
    api_db.add_row_v2(conn,"CLASS_EL_SUMMARY_MAPPING_METHOD",[id,str(class_id),str(currrentFrameData._frame_id),str(currrentFrameData._emotion["Angry"]),\
                                                                            str(currrentFrameData._emotion["Disgusted"]),str(currrentFrameData._emotion["Fearful"]),str(currrentFrameData._emotion["Happy"]),\
                                                                            str(currrentFrameData._emotion["Neutral"]),str(currrentFrameData._emotion["Sad"]),str(currrentFrameData._emotion["Surprised"]),\
                                                                            str(currrentFrameData._attention["focused"]),str(currrentFrameData._attention["distracted"]),\
                                                                            str(currrentFrameData._engagement_level_mapping_method["disengagement"]),str(currrentFrameData._engagement_level_mapping_method["low engagement"]),\
                                                                            str(currrentFrameData._engagement_level_mapping_method["medium engagement"]),str(currrentFrameData._engagement_level_mapping_method["high engagement"]),\
                                                                            str(currrentFrameData._engagement_level_mapping_method["strong engagement"])])

if __name__=="__main__":
    pass