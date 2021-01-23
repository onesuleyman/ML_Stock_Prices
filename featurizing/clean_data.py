""" Contain classes to clean data """

#import library
import pandas as pd 
import numpy as np 
import os 


class DataCleaner(object):

    def __init__(self, base_folder):
        self.base_folder = base_folder