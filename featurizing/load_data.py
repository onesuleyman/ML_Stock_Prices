""" contins classes to load the data """

#import library
import pandas as pd 
import numpy as np 
import os 

class DataLoader(object):

    def __init__(self, base_folder):
        self.base_folder = base_folder

    def load_data(self):
        #import data
        df = pd.read_csv(self.base_folder)
        
        return df
        