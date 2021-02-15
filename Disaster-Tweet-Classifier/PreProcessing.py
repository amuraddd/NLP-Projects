"""
PRE-PROCESS DATA
"""
import pandas as pd
import numpy as np
import Config

#============clean text from
def clean_text(dataframe, cols=Config.COLS_TO_CLEAN):
    """
    Apply regex to clean text.
    Parameters:
    datafarme: dataframe
    cols: columns which require textual cleaning
    Returns a cleaned dataframe
    """
    dataframe = dataframe.copy()
    for col in cols:
        if col=='keyword':
            dataframe[col] =  dataframe[col].str.replace(Config.KEYWORD_REGEX, ' ')
        if col=='location':
            dataframe[col] =  dataframe[col].str.replace(Config.LOCATION_REGEX, '')
        if col=='text':
            dataframe[col] =  dataframe[col].str.replace(Config.TEXT_REGEX, '')
    return dataframe

def replace_missing(dataframe):
    """
    Replace missing values for categorical variables by creating a missing category.
    Parameters:
    dataframe: dataframe with columns containing missing values
    Returns a dataframe with imputed values.
    """
    dataframe = dataframe.copy()
    cols_with_missing_values = [col for col in dataframe.columns if dataframe[col].isna().sum()>0]
    for col in cols_with_missing_values:
        if dataframe[col].dtype=='O':
            dataframe[col] = dataframe[col].fillna(col+'_missing')
    return dataframe

#======Create rare labels for values under a certain threshold
def find_frequent_labels(dataframe, var, pct, response=Config.TARGET_VALUE):
    """
    Find labels in categorical variables which are below the percentage threshold
    and return the indices of those labels.
    This function is used inside create_rare_label function.
    """
    dataframe = dataframe.copy()
    tmp = dataframe.groupby(var)[response].count()/len(dataframe)

    return tmp[tmp > pct].index

def create_rare_label(dataframe, cat_vars=Config.CAT_COLS, rare_pct=Config.RARE_PCT):
    """
    Take the indices of the rare labels and replace them with Rare.
    Return the dataframe with replaced rare labels.
    """
    for var in cat_vars:
        frequent_labels = find_frequent_labels(dataframe, var, rare_pct)
        dataframe[var] = np.where(dataframe[var].isin(frequent_labels), dataframe[var], 'Rare_'+var)

    return dataframe
