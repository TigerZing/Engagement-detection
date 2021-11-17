import sqlite3 as lite
import os
import os.path as osp
import sys
from sqlite3 import Error

# Read all datarow in table
def extract_all_row(conn, name_table):
    cur = conn.cursor()
    cmd_str = 'SELECT * FROM '+str(name_table)
    cur.execute(cmd_str)
    row = cur.fetchall()
    cur.close()
    return row

# Add row into table
def add_row(conn, name_table, columns, values):
    cur = conn.cursor()
    #print("UPDATE "+str(name_table)+" SET "+str(changed_column)+"='"+str(change_value)+"' WHERE "+str(condition_column)+"='"+str(condition_value)+"';")
    cur.execute("INSERT INTO "+str(name_table)+" ("+str(columns)+") VALUES ("+str(values)+");")
    conn.commit()
    cur.close()

def add_row_v2(conn, name_table, values):
    cur = conn.cursor()
    str_ = ""
    for idx, value in enumerate(values):
        if isinstance(value, int) or value == 'NULL':
            str_+=str(value)
        elif isinstance(value, str):
            str_+="'"+str(value)+"'"
        if idx != (len(values)-1):
            str_+=","
    # run command
    cur.execute("INSERT INTO "+name_table+" VALUES("+str_+")")
    conn.commit()
    cur.close()

# Add specific value into specific column by id
def update_specificValue(conn, name_table, changed_column, change_value, condition_column, condition_value):
    cur = conn.cursor()
    #print("UPDATE "+str(name_table)+" SET "+str(changed_column)+"='"+str(change_value)+"' WHERE "+str(condition_column)+"='"+str(condition_value)+"';")
    cur.execute("UPDATE "+str(name_table)+" SET "+str(changed_column)+"='"+str(change_value)+"' WHERE "+str(condition_column)+"='"+str(condition_value)+"';")
    conn.commit()
    cur.close()

# Reset reset_table
def reset_table(conn, name_table):
    cur = conn.cursor()
    cur.execute('DELETE FROM '+str(name_table))
    conn.commit()
    cur.close()

def intial_tables(conn):
    cur = conn.cursor()
    cur.execute("CREATE TABLE CLASS(ID INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, Id_class TEXT, Name_class TEXT, Lecturer TEXT, Total_student INT, Time_begin TEXT, Time_end TEXT)")
    cur.execute("CREATE TABLE STUDENT(Id integer primary key autoincrement, Id_student TEXT, Id_class TEXT, Name_student TEXT, Gender TEXT)")
    cur.execute("CREATE TABLE STUDENT_EL_MAPPING_METHOD(Id integer primary key autoincrement, Id_student TEXT, Id_class TEXT, Index_frame INT, Face_box TEXT, \
                                                        Face_id TEXT, Emotion TEXT, Attention TEXT, Engagement_level TEXT)")
    cur.execute("CREATE TABLE STUDENT_EL_KES_METHOD(Id integer primary key autoincrement, Id_student TEXT, Id_class TEXT, Index_frame INT,Face_box TEXT,\
                                                    Face_id TEXT, Emotion TEXT, Emotion_weight FLOAT, Eye_gaze_weight FLOAT, Concentration_index FLOAT,Engagement_level TEXT)")
    ### class summary
    cur.execute("CREATE TABLE CLASS_EL_SUMMARY_MAPPING_METHOD(Id integer primary key autoincrement, Id_class TEXT, Index_frame INT, \
                                                            Angry INT, Disgusted INT, Fearful INT, Happy INT, Neutral INT, Sad INT, Surprised INT, \
                                                            Focused INT, Distracted INT,\
                                                            Disengagement INT, Low_engagement INT, Medium_engagement INT, High_engagement INT, Strong_engagement INT)")
    cur.execute("CREATE TABLE CLASS_EL_SUMMARY_KES_METHOD(id integer primary key autoincrement, Id_class TEXT, Index_frame INT,\
                                                            Angry INT, Disgusted INT, Fearful INT, Happy INT, Neutral INT, Sad INT, Surprised INT,\
                                                            Disengagement INT, Engagement INT, High_engagement INT)")
    conn.commit()
    cur.close()

if __name__=="__main__":
    conn = lite.connect("/home/hont/Engagement_Detection/20210731.db")
    intial_tables(conn)
    conn.close()
    

        