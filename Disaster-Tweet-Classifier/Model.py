"""
Model
"""
import pandas as pd
import numpy as np
import Config
# from EDA import EDA
from LoadData import read_data
from FeatureEngineering import pre_process_engineer

data = read_data(Config.PATH_TO_DATA, Config.SQL_QUERY, Config.DATA_INDEX)
data = pre_process_engineer(data)
print(data.head())
