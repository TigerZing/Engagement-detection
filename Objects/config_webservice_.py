import os
import sys
import cv2
import numpy as np
import os.path as osp
import yaml
import io


class Config_Webservice:
    def __init__(self):
        # basic configs
        self.database_path = None
        self.reset_analytic_tables = None
        self.write2File = None
      
    # import info to object [Config]
    def import_config(self, parsed_config):
        # basic configs
        self.database_path = parsed_config["basic_config"]["database_path"]
        self.reset_analytic_tables = parsed_config["basic_config"]["reset_analytic_tables"]
        self.write2File = parsed_config["basic_config"]["write2File"]
        return self