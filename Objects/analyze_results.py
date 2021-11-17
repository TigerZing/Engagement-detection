from sqlite3.dbapi2 import Cursor, connect
import pandas as pd
from pandas import read_csv
from pandas.plotting import table
import dataframe_image as dfi
from sqlalchemy import create_engine
import sqlite3 as lite
import numpy as np
from DATABASE_API.connect_database import *
import matplotlib.pyplot as plt
import seaborn as sns;
import plotly.express as px
import random
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#Connect with data base
def Database_connection(data_path):
    conn = None
    try:
        conn = lite.connect(data_path)
    except Error as e:
        print(e, 'Woah Woah')
    return conn

#Read data frame
def read_df(conn, table_name):
    data_frame          = pd.read_sql_query("SELECT * FROM " + table_name , conn)
    return data_frame

#Define the range of each interval
def define_interval(df, No_interval):
    eof = int(round((max(df.Index_frame)//No_interval))+ 1)
    intervals = []
    for idx  in range(1,No_interval+1):
        intervals.append(idx * eof)
    intervals = np.array(intervals, dtype = np.int)
    return intervals, eof

#Assign each row in the datset with its interval
def label_interval(row, intervals,eof):
    temp_count = 0
    for index in range(len(intervals)):
        if row['Index_frame'] <= intervals[temp_count]:
            return index + 1
        else: temp_count += 1

#Define the score for each engagement level
def label_EL(row):
    if row['Engagement_level'] == 'strong engagement':
        return 4
    if row['Engagement_level'] == 'high engagement':
        return 3
    if row['Engagement_level'] == 'medium engagement':
        return 2
    if row['Engagement_level'] == 'low engagement':
        return 1
    if row['Engagement_level'] == 'disengagement':
        return 0
#Analyze the engagement level over the class.
def student_analyze_interval(student_EL,intervals, eof):

    student_EL['interval'] = student_EL.apply(lambda row: label_interval(row, intervals,eof), axis = 1)
    student_EL['interval'] = student_EL['interval'].astype(int)
    student_EL.to_csv('2020_08_11.csv')
    ### first return --- statistic information
    #statistic_inf = student_EL[['Engagement_level','Emotion', 'Attention']].describe().to_csv('summary.csv')
    statistic_inf = student_EL[['Engagement_level','Emotion', 'Attention']].describe()
    ### Display the distribution of student engagement level, student emotion and student concentration
    filter_value   = ["Engagement_level", "Emotion", "Attention"]
    piechart_fig   = make_subplots(rows = 1, cols = 3,specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])
    for i, value in enumerate(filter_value):
        EL_summary = student_EL[value].value_counts()
        piechart_fig.add_trace(go.Pie(values = EL_summary, labels = EL_summary.keys(),title = value) ,row=1, col=i+1)
        piechart   = piechart_fig.update_layout(title_text="Side By Side Subplots")
        #piechart_fig.write_html('piechart.html')
        piechart.show()
    

    ###Engagement Analytic for whole class in 10 interval
    counted_EL               = student_EL[['Id_student','Engagement_level','interval']]
    counted_EL['counted_EL'] = counted_EL.groupby([ 'Id_student','interval','Engagement_level'])['Engagement_level'].transform('count')
    counted_EL               = counted_EL.sort_values(by = 'Id_student', ascending = True)
    counted_EL               = counted_EL.drop_duplicates()
    maxes                    = counted_EL.groupby(['Id_student','interval'])['counted_EL'].idxmax()
    filtered_maxes           = counted_EL.loc[maxes]
    filtered_maxes           = filtered_maxes.sort_values(by = 'interval',ascending = True)
    filtered_maxes.reset_index()

    #make the interval stack bar by take the counted maximum EL group by interval
    count_EL_interval                      = filtered_maxes[['Engagement_level', 'interval']]
    count_EL_interval['count_by_interval'] = count_EL_interval.groupby(['interval','Engagement_level'])['Engagement_level'].transform('count')
    count_EL_interval.drop_duplicates()
    ELs                                    = count_EL_interval['Engagement_level'].unique()
    interval_EL_table                      = pd.DataFrame(columns=ELs)
    EL_arr = []
    for index, EL in enumerate(ELs):   
        engagement_summary = count_EL_interval[count_EL_interval['Engagement_level'] == EL].sort_values(by = 'interval', ascending=True)
        engagement_summary = engagement_summary.drop_duplicates()
        engagement_summary = engagement_summary.rename(columns={'count_by_interval':EL})
        engagement_summary = engagement_summary.drop(columns=['Engagement_level'])
        engagement_summary = engagement_summary.reset_index(drop = True)
        engagement_summary = engagement_summary.set_index('interval')
        EL_arr.append(engagement_summary[EL])
    EL_table = pd.concat(EL_arr, axis = 1)


    #engagement_chart_placeholder = st.empty()
    stack_barchart       = px.bar(EL_table, 
                template          = "none",
                x                 =  EL_table.index,
                y                 = [c for c in EL_table.columns],
                color_discrete_map= {'strong engagement': '#C7DBFF', 'medium engagement': '#6FA4FF', 'high engagement': '#8DB6FF', 'low engagement': '#5291FF', 'disengagement': '#bf360c'},
                labels            = {'value':'Number of student', 'x':'Timeline', 'variable':'Engagement level'},
                height            = 380
                )
    stack_barchart.update_layout(barmode='stack',font=dict(size=16), margin=dict(t=10),plot_bgcolor='rgba(0,0,0,0)',hovermode="x")
    stack_barchart.show()

    ####Concentration analytic for whole class in 10 interval
    counted_CI               = student_EL[['Id_student','Attention','interval']]
    counted_CI['counted_CI'] = counted_CI.groupby([ 'Id_student','interval','Attention'])['Attention'].transform('count')
    counted_CI               = counted_CI.sort_values(by = 'Id_student', ascending = True)
    counted_CI               = counted_CI.drop_duplicates()
    maxes                    = counted_CI.groupby(['Id_student','interval'])['counted_CI'].idxmax()
    filtered_CI_maxes        = counted_CI.loc[maxes]
    filtered_CI_maxes        = filtered_CI_maxes.sort_values(by = 'interval',ascending = True)
    filtered_CI_maxes.reset_index()

    #make the interval stack bar by take the counted maximum EL group by interval
    count_CI_interval                      = filtered_CI_maxes[['Attention', 'interval']]
    count_CI_interval['count_by_interval'] = count_CI_interval.groupby(['interval','Attention'])['Attention'].transform('count')
    count_CI_interval.drop_duplicates()
    CIs = count_CI_interval['Attention'].unique()

    # interval_CI_table = pd.DataFrame(columns=CIs)
    CI_arr = []
    for index, CI in enumerate(CIs): 
        print(CI)  
        CI_summary = count_CI_interval[count_CI_interval['Attention'] == CI].sort_values(by = 'interval', ascending=True)
        CI_summary = CI_summary.drop_duplicates()
        CI_summary = CI_summary.rename(columns={'count_by_interval':CI})
        CI_summary = CI_summary.drop(columns=['Attention'])
        CI_summary = CI_summary.reset_index(drop = True)
        CI_summary = CI_summary.set_index('interval')
        CI_arr.append(CI_summary[CI])
    CI_table = pd.concat(CI_arr, axis=1)
    CI_table

    data_ConcentrationData = CI_table
    CI_area_chart          = px.area(data_ConcentrationData, 
                template   = "none",
                x          = data_ConcentrationData.index,
                y          = [c for c in data_ConcentrationData.columns], 
                height     = 380,
                labels     = {'value':'Number of student', 'x':'Timeline', 'variable':'Concentration level'},
                )
    CI_area_chart.update_layout(barmode='stack',font=dict(size=16), margin=dict(t=10),plot_bgcolor='rgba(0,0,0,0)')
    CI_area_chart.show()
    
    return statistic_inf,piechart,stack_barchart,CI_area_chart
    
#Analyze the engagement level for each student.
def individual_analyze_interval(student_EL):
    EL_group = student_EL.groupby(['Id_student', 'interval'])
    counted_EL = EL_group.Engagement_level.value_counts().to_frame(name = 'counted_EL').reset_index()
    student_EL_dict = {g: d for g, d in counted_EL.groupby('Id_student')}
    student_arr = []
    print(student_EL_dict.keys())
    for student in student_EL_dict.keys():
        fig = px.bar(student_EL_dict[student], x = 'interval', y = 'counted_EL', color='Engagement_level', barmode = 'group', title = ('Engagement level in class of student ' + str(student)),
        color_discrete_map={'strong engagement': '#C7DBFF', 'medium engagement': '#6FA4FF', 'high engagement': '#8DB6FF', 'low engagement': '#5291FF', 'disengagement': '#bf360c'})
        fig.show()
        fig.write_html(student + '.html')
        dict_ = {"student_id":student, "figure":fig}
        student_arr.append(dict_)
    return student_arr

def main(conn, num_interval):
    ###path database
    #data_path   = "DATABASE/20210731.db"
    #conn        = Database_connection(data_path)
    #No_interval = 10

    ###list of table in databse
    class_inf       = "CLASS"
    student         = "STUDENT"
    student_mapping = "STUDENT_EL_MAPPING_METHOD"
    class_mapping   = "CLASS_EL_SUMMARY_MAPPING_METHOD"
    student_kes     = "STUDENT_EL_KES_METHOD"
    class_kes       = "CLASS_EL_SUMMARY_KES_METHOD"
    user_id         = "user_account"

    ### read table into datframe
    student_EL = read_df(conn,student_mapping)
    intervals, eof = define_interval(student_EL, num_interval)
    ### analyze the summary of student engagement table
    statistic_inf, piechart, stack_barchart, CI_area_chart = student_analyze_interval(student_EL,intervals, eof)
    student_arr = individual_analyze_interval(student_EL)

if __name__ == "__main__":
    main()