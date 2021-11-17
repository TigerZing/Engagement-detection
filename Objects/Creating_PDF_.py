import DATABASE_API
from fpdf import FPDF, HTMLMixin
import datetime
import glob
import sys
import codecs

PATH_IMAGE = "/home/ra/Documents/Huyen/Engagement_Detection_Framework/Output/Pdf_file/"

#using the fpdf library to create pdf file from image and text
#https://pyfpdf.readthedocs.io/en/latest/reference/set_auto_page_break/index.html

class PDF(FPDF,HTMLMixin):
    pass
    

    def initialize(self):
        self.set_auto_page_break(1,margin=15) #auto insert page break, otherwise: 0
        self.set_margins(left=10,top=17,right=10)
        self.add_page()
        self.set_title(title="CLASS SUMMARY REPORT")

        # width of the pdf file
        self.w = self.w-self.l_margin - self.r_margin

        #add logo
        self.image(PATH_IMAGE + "logo_login.png",x=self.w/2.3,y=3,w=44,h=22)
    
        #set title and datetime for the summary

        self.add_text("\n\nCLASS SUMMARY REPORT",w=self.w, style="B",fontsize=11,align="C")
        self.add_text("Date: "+str(datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")),w=self.w,align="C",fontsize=8)
        #self.add_text("\n")

    def add_image(self,image,x=10,y=None,width=130,height=80):
        #add image into pdf file
        self.image(image,x=x,y=y, w=width, h=height)
        self.multi_cell(w=1,h=3,txt="\n\n")
        
       
    def add_text(self,text,w=150, font = "Arial", style = "", fontsize = 8,align="J"):
        self.set_font(font, style, fontsize)
        self.multi_cell(w=w,h=5,txt=text+"\n",border=0,align=align)


    def footer(self,content="Research Center for Computing and Multimedia Studies, Hosei University", page_number = True):
        #this function is automatically generate by add_page func of FPDF library
        #if you want to change the content and page_number, Please edit in this func

        # Go to 1.5 cm from bottom
        self.set_y(-5)
        # Select Arial italic 8
        self.set_font('Arial', 'I', 6) 
        # Print centered page number
        if content:
            self.set_text_color(0, 0, 0)
            self.cell(180, 0, content, 0, 0 , 'C')
        if page_number:
            self.set_text_color(0, 0, 0)
            self.cell(10, 0, '%s' % self.page_no(), 0, 0, 'R')


    def add_HTML(self, html_content):
        self.write_html(html_content)

    def add_class_information(self,class_ID, class_name, lecturer,len_video,fps,resolution,num_students,num_intervals,time_begin,engagement_level, emotion_type, concentration_type, w_b_table=20):
        '''
        This function add the overview information:
        it includes 4 tables
        w_b_table: the width between each table

        '''
        self.set_y(self.get_y()+2)
        self.set_font('Arial', 'B', 8)
        self.cell(w = self.w/2,h=7,txt = "Basic information\n",align = "C",ln=0)
        self.cell(w = self.w/2,h=7,txt ="Overview analytics\n",align="C",ln=1)
        #set line width for each table
        self.set_line_width(0.1)
        self.set_font('Arial', '', 6)


        ######## add class information
        
        y = self.y

        width_column1 = self.get_string_width("Class Name") +3
        width_column2 = max(self.get_string_width(class_ID), self.get_string_width(class_name),self.get_string_width(lecturer)) + 3

        self.cell(w=width_column1,h=6,txt = "Course ID",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = class_ID,border=1,ln=1,align = "L")
        self.cell(w=width_column1,h=6,txt = "Class Name",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = class_name,border=1,ln=1,align = "L")
        self.cell(w=width_column1,h=6,txt = "Lecturer",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = lecturer,border=1,ln=0,align = "L")

        x= self.x + w_b_table

        ########## add video information
        width_column1 = self.get_string_width("Length of video") +3
        width_column2 = max(self.get_string_width(len_video), self.get_string_width(fps),self.get_string_width(resolution)) + 3

        self.set_xy(x,y)
        self.cell(w=width_column1,h=6,txt = "Length of video",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = len_video,border=1,ln=2,align = "L")

        self.set_x(x)
        self.cell(w=width_column1,h=6,txt = "FPS",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = fps,border=1,ln=2,align = "L")

        self.set_x(x)
        self.cell(w=width_column1,h=6,txt = "Resolution",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = resolution,border=1,ln=0,align = "L")


        x= self.x + w_b_table 
        # add processing information
        width_column1 = self.get_string_width("Total students") +3
        width_column2 = max(self.get_string_width(num_students), self.get_string_width(num_intervals),self.get_string_width(time_begin)) + 3

        self.set_xy(x,y)
        self.cell(w=width_column1,h=6,txt = "Total students",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = num_students,border=1,ln=2,align = "L")

        self.set_x(x)
        self.cell(w=width_column1,h=6,txt = "Num_intervals",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = num_intervals,border=1,ln=2,align = "L")

        self.set_x(x)
        self.cell(w=width_column1,h=6,txt = "Time begin",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = time_begin,border=1,ln=0,align = "L")

        #add overview analytics
        
        x= self.x + w_b_table 
        # add processing information
        width_column1 = self.get_string_width("Concentration type") +3
        width_column2 = max(self.get_string_width(engagement_level), self.get_string_width(emotion_type),self.get_string_width(concentration_type)) + 3

        self.set_xy(x,y)
        self.cell(w=width_column1,h=6,txt = "Engagement level",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = engagement_level,border=1,ln=2,align = "L")

        self.set_x(x)
        self.cell(w=width_column1,h=6,txt = "Emotion type",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = emotion_type,border=1,ln=2,align = "L")

        self.set_x(x)
        self.cell(w=width_column1,h=6,txt = "Concentration type",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = concentration_type,border=1,ln=2,align = "L")


    def add_student_information(self,Student_Name,engagement_level,emotion, concentration, First_distrated_time, second_distracted_time, image = PATH_IMAGE + "student_icon.png",result_image = None):
        """
        add student information in part 2
        student Name: name in the pdf file
        the table include -------------------------------------------
                          |engagement level| medium engagement (72%) |
                          -------------------------------------------
                          |Emotion         | Happy (58%)             |
                          -------------------------------------------
                          |Concentration   | focused                 |
                          -------------------------------------------
                         
        """
        
        self.set_y(self.get_y()+5)
        self.add_image(image,width=15,height=15)
        x,y= self.x+20,self.y-21
        self.set_font('Arial', 'B', 8)
        self.set_y(self.get_y()-3)
        self.cell(w=self.get_string_width(Student_Name)+3,h=4,txt = Student_Name,ln=0,align = "L")
        self.set_line_width(0.1)
        self.set_font('Arial', '', 6)

        
        width_column1 = self.get_string_width("Most Distracted Analytic") +3
        width_column2 = self.get_string_width("2nd Distracted Time: 00h00m00s") +3
        
        self.set_xy(x,y)
        self.cell(w=width_column1,h=6,txt = "Overall Engagement",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = engagement_level,border=1,ln=2,align = "L")

        self.set_x(x)
        self.cell(w=width_column1,h=6,txt = "Overall Emotion",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = emotion,border=1,ln=2,align = "L")

        self.set_x(x)
        self.cell(w=width_column1,h=6,txt = "Overall Concentration",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt = concentration,border=1,ln=2,align = "L")
        self.set_x(x)
        self.cell(w=width_column1,h=12,txt = "Most Distracted Analytic",border=1,ln=0,align = "L")
        self.cell(w=width_column2,h=6,txt ="1st Distracted Time: " + First_distrated_time ,border=1,ln=2,align = "L")
        self.cell(w=width_column2,h=6,txt ="2nd Distracted Time: " + second_distracted_time ,border=1,ln=2,align = "L")

        self.add_image(result_image,x=self.x +40,y = y-5,width=105,height=75)
        self.set_y(self.get_y()+28)
        

def main():

    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.initialize()
    
    pdf.add_text("1. Overview\n",style="B",fontsize=10)
    #pdf.line(pdf.x, pdf.y, pdf.x+pdf.w, pdf.y)
    #Class information
    pdf.add_class_information("Signal1","Signal Coding","Admin","0h:13m:48s","25","1280x720","5","10","12","medium engagement(47%)","Happy(45%)","focused(84%)")

    pdf.set_xy(pdf.l_margin, pdf.get_y()+10)
    pdf.set_font('Arial', 'B', 8)
    pdf.cell(w = pdf.w/2,h=7,txt = "Engagement Timeline Visualization",align = "C",ln=0)
    pdf.cell(w = pdf.w/2,h=7,txt ="Attention Timeline Visualization",align="C",ln=1)
 
    # get the position y for concentration visualization
    y = pdf.get_y()
    
    pdf.set_xy(pdf.l_margin,y)
    img =PATH_IMAGE+"engagement_timeline_visualization.png"
    pdf.add_image(img, y=y,height=50 ,width=pdf.w/2)

    #concentration index
    pdf.set_x(y)
    img = PATH_IMAGE+"attention_timeline_visualization.png"
    pdf.add_image(img,x=pdf.w/2+10, y=y,height=50,width=pdf.w/2)

    ######################
    ##################### visualization in pie chart
    pdf.set_xy(pdf.l_margin, pdf.get_y()+40)
    pdf.set_font('Arial', 'B', 8)
    pdf.cell(w = pdf.w/3,h=7,txt = "Emotion Information",align = "C",ln=0)
    pdf.cell(w = pdf.w/3,h=7,txt ="Engagement Visualization",align="C",ln=0)
    pdf.cell(w = pdf.w/3,h=7,txt ="Concentration Visualization",align="C",ln=1)

    # get the position y for concentration visualization
    y = pdf.get_y()
    
    #############add pie chart
    #emotion
    pdf.set_xy(pdf.l_margin,y)
    img =PATH_IMAGE+"emotion_information.png"
    pdf.add_image(img, y=y,height=50 ,width=pdf.w/3-5)

    #engagement visualization
    pdf.set_x(y)
    img = PATH_IMAGE+"engagement_visualization.png"
    pdf.add_image(img,x=pdf.w/3+7, y=y,height=50,width=pdf.w/3-5)

     #concentration
    pdf.set_x(y)
    img = PATH_IMAGE+"concentration_visualization.png"
    pdf.add_image(img,x=2*pdf.w/3+7, y=y,height=50,width=pdf.w/3-5)
    ########## end pie chart

    
    pdf.set_xy(pdf.l_margin,pdf.get_y()+40)
    #page 2: List of student
    pdf.add_text("2. Engagement Learning Analytics",style="B",fontsize=10)

    listStudent = glob.glob(PATH_IMAGE+"student/*.jpeg")
    listStudent.sort()
    for student in listStudent:
        student_ID = student.split("/")[-1][:-5]
        pdf.add_student_information(student_ID,"medium engagement (72%)","Happy (58%)","focused (98%)",result_image=student)

    pdf.output('Summary_of_student_engagement.pdf','F')
    print("Automatically genenrate pdf file...")

if __name__ =="__main__":
    main()