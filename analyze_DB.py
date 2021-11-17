# from sqlite3.dbapi2 import Cursor, connect
# import pandas as pd
# from pandas import read_csv
# from pandas.plotting import table
# import dataframe_image as dfi
# from sqlalchemy import create_engine
# import sqlite3 as lite
# import numpy as np
# from DATABASE_API.connect_database import *
# import matplotlib.pyplot as plt
# import seaborn as sns;
# import plotly.express as px
# import random
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import subprocess
import pandas as pd
import sqlite3 as lite
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
from Database import connect_db
st.set_page_config(page_title = 'Engagement Detection System', layout="wide")

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

def EL_sorter(df):
    sorter = pd.DataFrame({'Engagement level': ['disengagement', 'low engagement', 'medium engagement', 'high engagement', 'strong engagement']})
    sort_mapping = sorter.reset_index().set_index('Engagement level')
    df['EL_num'] = df['Engagement_level'].map(sort_mapping['index'])
    df           = df.sort_values(by  = 'EL_num', ascending= False)
    return df

#Analyze the engagement level over the class.
def statistic_information(student_EL):
    #Class analyzing
    class_info = student_EL[['Engagement_level','Emotion', 'Attention']].describe().T
    class_info['Percentage'] = class_info['freq']/class_info['count']*100
    class_info['Percentage'] = class_info['Percentage'].apply(lambda x: round(x, 2))
    class_summary = class_info[['top','Percentage']].to_dict('index')
    #print(class_summary)

    #Each student analyzing
    student_dict = {g: d for g, d in student_EL.groupby('Id_student')}
    student_summary = {}
    for student in student_dict:
        summary =pd.DataFrame(student_dict[student][['Engagement_level','Emotion', 'Attention']].describe().T)
        summary['Percentage'] = summary['freq']/summary['count']*100
        summary['Percentage'] = summary['Percentage'].apply(lambda x: round(x, 2))
        summary_transpose = summary.T
        sub_dict = summary[['top','Percentage']].to_dict('index')
        #print(sub_dict)
        dict_ = {student: sub_dict}
        student_summary.update(dict_)
        #print(student,summary[['top','Percentage']])
    return class_summary, student_summary


def student_analyze_interval(student_EL,intervals, eof,fps):

    student_EL['interval'] = student_EL.apply(lambda row: label_interval(row, intervals,eof), axis = 1)
    student_EL['interval'] = student_EL['interval'].astype(int)
    #student_EL.to_csv('2020_08_11.csv')
    time_line = intervals/(fps*60)
    time_line = pd.to_datetime(time_line, unit= 'm').strftime("%H:%M:%S")
    interval_bins = np.arange(start = 0, stop = len(time_line)+1, step = 1)

    ### first return --- statistic information
    #statistic_inf = student_EL[['Engagement_level','Emotion', 'Attention']].describe()

    ### Display the distribution of student engagement level, student emotion and student concentration
    filter_value   = ["Engagement_level", "Emotion", "Attention"]

    Emotion_level_Data = student_EL["Emotion"].value_counts(sort=False)
    emotion_color_discrete_map= {'Angry': '#EE6666', 'Disgusted': '#5470C6', 'Fearful': '#73C0DE', 'Happy': '#FAC858', 'Neutral': '#3BA272', 'Sad':'#91CC75', 'Surprised':'#FC8452'}
    list_data_EmotionLabel = list(Emotion_level_Data.keys())
    emotion_colors = []
    for emotion_type_ in list_data_EmotionLabel:
        emotion_colors.append(emotion_color_discrete_map[emotion_type_])
    
    #colors.append(color_discrete_map[emotion_.key()])
    #data_EmotionLabel = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    data_emotion = go.Pie(labels = Emotion_level_Data.keys(), values = Emotion_level_Data)
    
    layout = go.Layout(font=dict(size=14),autosize=False,width=500,height=400, 
            xaxis= go.layout.XAxis(linecolor = 'black', linewidth = 1, mirror = True),
            yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 3, mirror = True),
            margin=go.layout.Margin(l=30, r=30, b=30, t=30,pad = 4))
    fig_Emotion = go.Figure(data = [data_emotion] ,layout=layout)
    #colors = ['#EE6666', '#5470C6','#73C0DE','#FAC858','#3BA272','#91CC75','#FC8452']
    #colors = np.array([''] * len(crit), dtype = object)
    #for i in np.unique(Emotion_level_Data):
    #    colors[np.where(crit == i)] = color_dict[str(i)]
    fig_Emotion.update_traces(hoverinfo='label+value', textinfo='label+percent',marker=dict(colors=emotion_colors))
    # visualize figure
    #st.plotly_chart(fig_Emotion, use_container_width=True)

    Engagement_level_Data = student_EL["Engagement_level"].value_counts(sort=False)
    engagement_color_discrete_map = {'strong engagement':'#C7DBFF', 'medium engagement':'#6FA4FF', 'high engagement':'#8DB6FF', 'low engagement':'#5291FF', 'disengagement':'#bf360c'}
    list_data_EngagementLabel = list(Engagement_level_Data.keys())    
    engagement_colors = []
    for engagement_type_ in list_data_EngagementLabel:
        engagement_colors.append(engagement_color_discrete_map[engagement_type_])
        
    #data_EngagementLabel = ['strong engagement', 'medium engagement', 'high engagement', 'low engagement', 'disengagement']
    data_Engagement = go.Pie(labels = Engagement_level_Data.keys(), values = Engagement_level_Data)
    layout = go.Layout(font=dict(size=14),autosize=False,width=500,height=400, 
            xaxis= go.layout.XAxis(linecolor = 'black', linewidth = 1, mirror = True),
            yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 3, mirror = True),
            margin=go.layout.Margin(l=30, r=30, b=30, t=30,pad = 4))
    fig_Engagement = go.Figure(data = [data_Engagement] ,layout=layout)
    #colors = ['#C7DBFF', '#6FA4FF','#8DB6FF','#5291FF','#bf360c']
    fig_Engagement.update_traces(hoverinfo='label+value', textinfo='label+percent',marker=dict(colors=engagement_colors))
    # visualize figure
    #st.plotly_chart(fig_Engagement, use_container_width=True)

    CI_level_Data = student_EL["Attention"].value_counts(sort=False)
    CI_color_discrete_map = {'focused':'#8FBBD9', 'distracted':'#FFBF86'}
    list_data_CILabel = list(CI_level_Data.keys())    
    CI_colors = []
    for CI_type_ in list_data_CILabel:
        CI_colors.append(CI_color_discrete_map[CI_type_])


    #data_CI_level_DataLabel = ['focused', 'distracted']
    data_CI = go.Pie(labels = CI_level_Data.keys(), values = CI_level_Data)
    layout = go.Layout(font=dict(size=14),autosize=False,width=500,height=400, 
            xaxis= go.layout.XAxis(linecolor = 'black', linewidth = 1, mirror = True),
            yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 3, mirror = True),
            margin=go.layout.Margin(l=30, r=30, b=30, t=30,pad = 4))
    fig_CI = go.Figure(data = [data_CI] ,layout=layout)
    #colors = ['#8FBBD9', '#FFBF86']
    fig_CI.update_traces(hoverinfo='label+value', textinfo='label+percent',marker=dict(colors=CI_colors))
    # visualize figure
    #st.plotly_chart(fig_CI, use_container_width=True)

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
    count_EL_interval                      = EL_sorter(count_EL_interval)
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

    EL_table['time_line'] = time_line
    #engagement_chart_placeholder = st.empty()
    stack_barchart       = px.bar(EL_table, 
                template          = "none",
                x                 =  EL_table.time_line,
                y                 = EL_table.columns,
                color_discrete_map= {'strong engagement': '#C7DBFF', 'medium engagement': '#6FA4FF', 'high engagement': '#8DB6FF', 'low engagement': '#5291FF', 'disengagement': '#bf360c'},
                labels            = {'value':'Number of student', 'x':'Timeline', 'variable':'Engagement level'},
                )
    stack_barchart.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = EL_table.time_line,
            dtick = 0.75
        )
    )

    ####Concentration analytic for whole class in 10 interval
    counted_CI               = student_EL[['Id_student','Engagement_level','interval','Attention']]
    counted_CI['counted_CI'] = counted_CI.groupby([ 'Id_student','interval','Engagement_level'])['Engagement_level'].transform('count')
    counted_CI               = counted_CI.sort_values(by = 'Id_student', ascending = True)
    counted_CI               = counted_CI.drop_duplicates()
    maxes                    = counted_CI.groupby(['Id_student','interval'])['counted_CI'].idxmax()
    filtered_CI_maxes        = counted_CI.loc[maxes]
    filtered_CI_maxes        = filtered_CI_maxes.sort_values(by = 'interval',ascending = True)
    filtered_CI_maxes        = filtered_CI_maxes.reset_index()
    filtered_CI_maxes.head()

    #make the interval stack bar by take the counted maximum EL group by interval
    count_CI_interval                      = filtered_CI_maxes[['Attention', 'interval']]
    count_CI_interval['count_by_interval'] = count_CI_interval.groupby(['interval','Attention'])['Attention'].transform('count')
    count_CI_interval.drop_duplicates()
    CIs = count_CI_interval['Attention'].unique()

    CI_arr = []
    for index, CI in enumerate(CIs): 
        CI_summary = count_CI_interval[count_CI_interval['Attention'] == CI].sort_values(by = 'interval', ascending=True)
        CI_summary = CI_summary.drop_duplicates()
        CI_summary = CI_summary.rename(columns={'count_by_interval':CI})
        CI_summary = CI_summary.drop(columns=['Attention'])
        CI_summary = CI_summary.reset_index(drop = True)
        CI_summary = CI_summary.set_index('interval')
        CI_arr.append(CI_summary[CI])
    CI_table = pd.concat(CI_arr, axis=1)
    CI_table['time_line'] = time_line
    padding_CI_row = pd.DataFrame({'focused':0,'distracted':0,	'time_line':0}, index = [0])
    CI_table = pd.concat([padding_CI_row, CI_table]).reset_index(drop = True)
    data_ConcentrationData = CI_table
    CI_area_chart          = px.area(data_ConcentrationData, 
                template   = "none",
                x          = data_ConcentrationData.time_line,
                y          = [c for c in data_ConcentrationData.columns], 
                labels     = {'value':'Number of student', 'x':'Timeline', 'variable':'Concentration level'},
                )
    
    return stack_barchart, CI_area_chart, fig_Emotion, fig_Engagement, fig_CI
    
def workplace_student_analyze_interval(student_EL,intervals, eof,fps):

    student_EL['interval'] = student_EL.apply(lambda row: label_interval(row, intervals,eof), axis = 1)
    student_EL['interval'] = student_EL['interval'].astype(int)
    #student_EL.to_csv('2020_08_11.csv')
    time_line = intervals/(fps*60)
    time_line = pd.to_datetime(time_line, unit= 'm').strftime("%H:%M:%S")
    interval_bins = np.arange(start = 0, stop = len(time_line)+1, step = 1)

    ### first return --- statistic information
    #statistic_inf = student_EL[['Engagement_level','Emotion', 'Attention']].describe()
    ### Display the distribution of student engagement level, student emotion and student concentration
    filter_value   = ["Engagement_level", "Emotion", "Attention"]

    Emotion_level_Data = student_EL["Emotion"].value_counts(sort=False)
    emotion_color_discrete_map= {'Angry': '#EE6666', 'Disgusted': '#5470C6', 'Fearful': '#73C0DE', 'Happy': '#FAC858', 'Neutral': '#3BA272', 'Sad':'#91CC75', 'Surprised':'#FC8452'}
    list_data_EmotionLabel = list(Emotion_level_Data.keys())
    emotion_colors = []
    for emotion_type_ in list_data_EmotionLabel:
        emotion_colors.append(emotion_color_discrete_map[emotion_type_])
    
    #colors.append(color_discrete_map[emotion_.key()])
    #data_EmotionLabel = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    data_emotion = go.Pie(labels = Emotion_level_Data.keys(), values = Emotion_level_Data)
    
    layout = go.Layout(font=dict(size=12),autosize=False, 
            xaxis= go.layout.XAxis(linecolor = 'black', linewidth = 1, mirror = True),
            yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 3, mirror = True))
    layout = go.Layout({"showlegend": False})
    fig_Emotion = go.Figure(data = [data_emotion] ,layout=layout)
    #colors = ['#EE6666', '#5470C6','#73C0DE','#FAC858','#3BA272','#91CC75','#FC8452']
    #colors = np.array([''] * len(crit), dtype = object)
    #for i in np.unique(Emotion_level_Data):
    #    colors[np.where(crit == i)] = color_dict[str(i)]
    fig_Emotion.update_traces(hoverinfo='label+value', textinfo='label+percent',marker=dict(colors=emotion_colors))
    # visualize figure
    #st.plotly_chart(fig_Emotion, use_container_width=True)

    Engagement_level_Data = student_EL["Engagement_level"].value_counts(sort=False)
    engagement_color_discrete_map = {'strong engagement':'#C7DBFF', 'medium engagement':'#6FA4FF', 'high engagement':'#8DB6FF', 'low engagement':'#5291FF', 'disengagement':'#bf360c'}
    list_data_EngagementLabel = list(Engagement_level_Data.keys())    
    engagement_colors = []
    for engagement_type_ in list_data_EngagementLabel:
        engagement_colors.append(engagement_color_discrete_map[engagement_type_])
        
    #data_EngagementLabel = ['strong engagement', 'medium engagement', 'high engagement', 'low engagement', 'disengagement']
    data_Engagement = go.Pie(labels = Engagement_level_Data.keys(), values = Engagement_level_Data)
    layout = go.Layout(font=dict(size=12),autosize=True,
            xaxis= go.layout.XAxis(linecolor = 'black', linewidth = 1, mirror = True),
            yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 3, mirror = True))
    layout = go.Layout({"showlegend": False})
    fig_Engagement = go.Figure(data = [data_Engagement] ,layout=layout)
    #colors = ['#C7DBFF', '#6FA4FF','#8DB6FF','#5291FF','#bf360c']
    fig_Engagement.update_traces(hoverinfo='label+value', textinfo='label+percent',marker=dict(colors=engagement_colors))
    # visualize figure
    #st.plotly_chart(fig_Engagement, use_container_width=True)

    CI_level_Data = student_EL["Attention"].value_counts(sort=False)
    CI_color_discrete_map = {'focused':'#8FBBD9', 'distracted':'#FFBF86'}
    list_data_CILabel = list(CI_level_Data.keys())    
    CI_colors = []
    for CI_type_ in list_data_CILabel:
        CI_colors.append(CI_color_discrete_map[CI_type_])


    #data_CI_level_DataLabel = ['focused', 'distracted']
    data_CI = go.Pie(labels = CI_level_Data.keys(), values = CI_level_Data)
    layout = go.Layout(font=dict(size=12),autosize=True, 
            xaxis= go.layout.XAxis(linecolor = 'black', linewidth = 1, mirror = True),
            yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 3, mirror = True),
            margin=go.layout.Margin(l=30, r=30, b=30, t=10,pad = 4))
    layout = go.Layout({"showlegend": False})
    fig_CI = go.Figure(data = [data_CI] ,layout=layout)
    #colors = ['#8FBBD9', '#FFBF86']
    fig_CI.update_traces(hoverinfo='label+value', textinfo='label+percent',marker=dict(colors=CI_colors))
    # visualize figure
    #st.plotly_chart(fig_CI, use_container_width=True)

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
    count_EL_interval                      = EL_sorter(count_EL_interval)
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

    EL_table['time_line'] = time_line
    #engagement_chart_placeholder = st.empty()
    stack_barchart       = px.bar(EL_table, 
                template          = "none",
                x                 =  EL_table.time_line,
                y                 = EL_table.columns,
                color_discrete_map= {'strong engagement': '#C7DBFF', 'medium engagement': '#6FA4FF', 'high engagement': '#8DB6FF', 'low engagement': '#5291FF', 'disengagement': '#bf360c'},
                labels            = {'value':'Number of student', 'x':'Timeline', 'variable':''},
                height=250,
                )
    stack_barchart.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = EL_table.time_line,
            dtick=1.0
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    ####Concentration analytic for whole class in 10 interval
    counted_CI               = student_EL[['Id_student','Engagement_level','interval','Attention']]
    counted_CI['counted_CI'] = counted_CI.groupby([ 'Id_student','interval','Engagement_level'])['Engagement_level'].transform('count')
    counted_CI               = counted_CI.sort_values(by = 'Id_student', ascending = True)
    counted_CI               = counted_CI.drop_duplicates()
    maxes                    = counted_CI.groupby(['Id_student','interval'])['counted_CI'].idxmax()
    filtered_CI_maxes        = counted_CI.loc[maxes]
    filtered_CI_maxes        = filtered_CI_maxes.sort_values(by = 'interval',ascending = True)
    filtered_CI_maxes        = filtered_CI_maxes.reset_index()
    filtered_CI_maxes.head()

    #make the interval stack bar by take the counted maximum EL group by interval
    count_CI_interval                      = filtered_CI_maxes[['Attention', 'interval']]
    count_CI_interval['count_by_interval'] = count_CI_interval.groupby(['interval','Attention'])['Attention'].transform('count')
    count_CI_interval.drop_duplicates()
    CIs = count_CI_interval['Attention'].unique()

    CI_arr = []
    for index, CI in enumerate(CIs): 
        CI_summary = count_CI_interval[count_CI_interval['Attention'] == CI].sort_values(by = 'interval', ascending=True)
        CI_summary = CI_summary.drop_duplicates()
        CI_summary = CI_summary.rename(columns={'count_by_interval':CI})
        CI_summary = CI_summary.drop(columns=['Attention'])
        CI_summary = CI_summary.reset_index(drop = True)
        CI_summary = CI_summary.set_index('interval')
        CI_arr.append(CI_summary[CI])
    CI_table = pd.concat(CI_arr, axis=1)
    CI_table['time_line'] = time_line
    padding_CI_row = pd.DataFrame({'focused':0,'distracted':0,	'time_line':0}, index = [0])
    CI_table = pd.concat([padding_CI_row, CI_table]).reset_index(drop = True)
    data_ConcentrationData = CI_table
    CI_area_chart          = px.area(data_ConcentrationData, 
                template   = "none",
                x          = data_ConcentrationData.time_line,
                y          = [c for c in data_ConcentrationData.columns], 
                labels     = {'value':'Number of student', 'x':'Timeline', 'variable':''},
                height=250,
                )
    CI_area_chart.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return stack_barchart, CI_area_chart, fig_Emotion, fig_Engagement, fig_CI
  
#Analyze the engagement level for each student.
def individual_analyze_interval(student_EL,intervals,fps):
    padding_row = {'Id_student':'0','time_line':'00:00:00','Engagement_level':'strong engagement','counted_EL':0}
    padding_row = pd.DataFrame([padding_row],index=[0])
    time_line = intervals/(fps*60)
    time_line = pd.to_datetime(time_line, unit= 'm').strftime("%H:%M:%S")
    interval_bins = np.arange(start = 0, stop = len(time_line)+1, step = 1)
    student_EL['time_line'] = pd.cut(student_EL['interval'], bins = interval_bins, labels = time_line)

    EL_group = student_EL.groupby(['Id_student','time_line'])
    counted_EL = EL_group.Engagement_level.value_counts().to_frame(name = 'counted_EL').reset_index()
    student_EL_dict = {g: d for g, d in counted_EL.groupby('Id_student')}
    student_arr = []
    disengagement_dict = {}
    for student in student_EL_dict.keys():
        student_EL_dict[student]  = pd.concat([padding_row,student_EL_dict[student]])
        student_EL_dict[student] = student_EL_dict[student].sort_values('time_line')\
                                    .reset_index(drop=True)
        student_EL_dict[student] = student_EL_dict[student].pivot(index = 'time_line',columns = 'Engagement_level', values = 'counted_EL')
        student_EL_dict[student] = student_EL_dict[student].reset_index()
        student_EL_dict[student]['previous_time'] = student_EL_dict[student]['time_line'].shift(1)
        if 'disengagement' in student_EL_dict[student].columns:
            disengagement_dict.update({student:student_EL_dict[student][['disengagement','previous_time','time_line']].rename(columns = {'disengagement':'score','time_line':'current_time'})
                                                                                                       .nlargest(2,'score').to_dict('records')})
        # student_EL_dict[student].reset_index(drop=True)
        # student_EL_dict[student].drop(columns = ['previous_time'])
        student_EL_dict[student] = student_EL_dict[student].reset_index(drop=True).set_index('time_line')
        student_EL_dict[student] = student_EL_dict[student].drop(columns = ['previous_time'])
        fig       = px.area( student_EL_dict[student], 
            template          = "none",
            x                 =   student_EL_dict[student].index,
            y                 = [c for c in  student_EL_dict[student].columns],
            color_discrete_map= {'strong engagement': '#C7DBFF', 'medium engagement': '#6FA4FF', 'high engagement': '#8DB6FF', 'low engagement': '#5291FF', 'disengagement': '#bf360c'},
            labels            = {'value':'Score', 'x':'Timeline', 'variable':'Engagement level'},
            height = 500,
            )
        dict_ = {"student_id":student, "figure":fig}
        student_arr.append(dict_)
    return student_arr, disengagement_dict

def analyze_db_main(conn, num_interval, fps):
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
    #student_EL.to_csv("student_EL.csv")
    intervals, eof = define_interval(student_EL, num_interval)
    ### analyze the summary of student engagement table

    stack_barchart, CI_area_chart, fig_Emotion, fig_Engagement, fig_CI = workplace_student_analyze_interval(student_EL, intervals, eof, fps)
    #st.plotly_chart(stack_barchart, use_container_width=True)
    #st.plotly_chart(CI_area_chart, use_container_width=True)
    student_arr, disengagement_dict= individual_analyze_interval(student_EL,intervals,fps) 

    class_summary, student_summary = statistic_information(student_EL)

    return stack_barchart, CI_area_chart, fig_Emotion, fig_Engagement, fig_CI, class_summary, student_summary, student_arr, disengagement_dict

def interface():
    database_conn = connect_db.create_connection("/home/hont/Engagement_Detection/Database/main.db")
    stack_barchart, CI_area_chart, fig_Emotion, fig_Engagement, fig_CI, class_summary, student_summary, student_arr, disengagement_dict = analyze_db_main(database_conn, 10, 25)
    database_conn.close()

    # interface
    ctop3, ctop1 = st.beta_columns((10, 10))
    mid1, mid2, mid3, _ = st.beta_columns((10, 10, 10, 5))
    
    with ctop1:
        #st.markdown("<h3 style='color:#0085ad; text-align:center;'> CONCENTRATION VISUALIZATION </h3>", unsafe_allow_html=True)
        concentration_chart_placeholder = st.empty()
        # if input_page_data["course_id"] =="":
        #     input_page_data["course_id"] = "N/A"
        # st.markdown('<p> <span style="color:#be0027;"> <strong>Class ID</strong></span>            :  {} </p>'.format(input_page_data["course_id"]), unsafe_allow_html=True)
        # if input_page_data["course_name"] =="":
        #     input_page_data["course_name"] = "N/A"
        # st.markdown('<p> <span style="color:#be0027;"> <strong>Class Name</strong></span>           :  {} </p>'.format(input_page_data["course_name"]), unsafe_allow_html=True)
        # st.markdown('<p> <span style="color:#be0027;"> <strong>Lecturer</strong></span>             :  {} </p>'.format(str(st.session_state.userFullname)), unsafe_allow_html=True)
        # if input_page_data["total_students"] =="":
        #     input_page_data["total_students"] = "N/A"
        # st.markdown('<p> <span style="color:#be0027;"> <strong>Total of student</strong></span>     :  {} </p>'.format(input_page_data["total_students"]), unsafe_allow_html=True)
        # if input_page_data["time_begin"] =="":
        #     input_page_data["time_begin"] = "N/A"
        # st.markdown('<p> <span style="color:#be0027;"> <strong>Time</strong></span>                 :  {}</p>'.format(input_page_data["time_begin"]), unsafe_allow_html=True)
        concentration_chart_placeholder.plotly_chart(CI_area_chart, use_container_width=True)
    with ctop3:
        # placeholder 
        #st.markdown("<h3 style='color:#0085ad; text-align:center;'> ENGAGEMENT VISUALIZATION </h3>", unsafe_allow_html=True)
        engagement_chart_placeholder = st.empty()
        #             # settings
        # fig = px.bar(st.session_state.data_Engagement, 
        #             template = "none",
        #             x =  st.session_state.timeline_data_Engagement,
        #             y = [c for c in st.session_state.data_Engagement.columns],
        #             color_discrete_map={'strong engagement': '#C7DBFF', 'medium engagement': '#6FA4FF', 'high engagement': '#8DB6FF', 'low engagement': '#5291FF', 'disengagement': '#bf360c'},
        #             labels={'value':'Number of student', 'x':'Timeline', 'variable':'Engagement level'},
        #             height=380
        #             )
        # fig.update_layout(barmode='stack',font=dict(size=16), margin=dict(t=10),plot_bgcolor='rgba(0,0,0,0)',hovermode="x")
        # visualize figure
        engagement_chart_placeholder.plotly_chart(stack_barchart, use_container_width=True)


    
    with mid2:
        #st.markdown("<h3 style='color:#0085ad; text-align:center;'> EMOTION </h3>", unsafe_allow_html=True)
        # placeholder
        emotion_chart_placeholder = st.empty()
        # data_EmotionLabel = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        # #data_EmotionScore = [12,18,30,10,5,15,10]
        # # settings
        # data = go.Pie(labels = data_EmotionLabel, values = st.session_state.data_EmotionScore)
        # layout = go.Layout(font=dict(size=14),autosize=False,width=400,height=400, 
        #         xaxis= go.layout.XAxis(linecolor = 'black', linewidth = 1, mirror = True),
        #         yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 3, mirror = True),
        #         margin=go.layout.Margin(l=50, r=50, b=100, t=10,pad = 4))
        # fig = go.Figure(data = [data] ,layout=layout)
    
        # colors = ['#EE6666', '#5470C6','#73C0DE','#FAC858','#3BA272','#91CC75','#FC8452']
        # fig.update_traces(hoverinfo='label+value', textinfo='label+percent',marker=dict(colors=colors))
        # visualize figure
        emotion_chart_placeholder.plotly_chart(fig_Emotion, use_container_width=True)
    
    with mid1:
        #st.markdown("<h3 style='color:#0085ad; text-align:center;'> ENGAGEMENT DISTRIBUTION </h3>", unsafe_allow_html=True)
        engagement_piechart_placeholder = st.empty()
        # # settings
        # fig = px.area(st.session_state.data_ConcentrationData, 
        #             template = "none",
        #             x = st.session_state.data_ConcentrationData.index,
        #             y = [c for c in st.session_state.data_ConcentrationData.columns], 
        #             height=380,
        #             labels={'value':'Number of student', 'x':'Timeline', 'variable':'Concentration level'},
        #             )
        # fig.update_layout(barmode='stack',font=dict(size=16), margin=dict(t=10),plot_bgcolor='rgba(0,0,0,0)')
        # # visualize figure
        engagement_piechart_placeholder.plotly_chart(fig_Engagement, use_container_width=True)
    #st.markdown("<h3 style='color:#0085ad; text-align:center;'> OUPUT VISUALIZATION </h3><br>", unsafe_allow_html=True)
    with mid3:
        #st.markdown("<h3 style='color:#0085ad; text-align:center;'> CONCENTRATION VISUALIZATION </h3>", unsafe_allow_html=True)
        concentration_piechart_placeholder = st.empty()
        # # settings
        # fig = px.area(st.session_state.data_ConcentrationData, 
        #             template = "none",
        #             x = st.session_state.data_ConcentrationData.index,
        #             y = [c for c in st.session_state.data_ConcentrationData.columns], 
        #             height=380,
        #             labels={'value':'Number of student', 'x':'Timeline', 'variable':'Concentration level'},
        #             )
        # fig.update_layout(barmode='stack',font=dict(size=16), margin=dict(t=10),plot_bgcolor='rgba(0,0,0,0)')
        # # visualize figure
        concentration_piechart_placeholder.plotly_chart(fig_CI, use_container_width=True)
    
    
if __name__ == "__main__":
    interface()