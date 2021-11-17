import os
import sys
import cv2
import numpy as np
import os.path as osp
import yaml
import io


class User_Account:
    def __init__(self):
        # basic configs
        self.username = None
        self.password = None
        self.email = None
        self.fullname = None
        self.date_login = None
        self.user_data_storage = None
      
    # import info to object [Config]
    def _intialize(self, username, password, email, fullname , date_login, user_data_storage):
        self.username = username
        self.password = password
        self.email = email
        self.fullname = fullname
        self.date_login = date_login
        self.user_data_storage = user_data_storage

    def _convert_dataRow(self, dataRow):
        self.username = dataRow[0]
        self.password = dataRow[1]
        self.email = dataRow[2]
        self.fullname = dataRow[3]
        self.date_login = dataRow[4]
        self.user_data_storage = dataRow[5]

