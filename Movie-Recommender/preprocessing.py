"""
Data Pre-processing
"""
import pandas as pd
import datetime
from datetime import datetime
import re
import numpy as np
import string
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def extract_features(value):
    """
    Take the description of each movies as the input value and apply the following operations:
        1. Tokenize the description.
        2. Remove stopwords
        3. Remove punctuation
        4. Lemmatize the words
        5. Append the cleaned description as words to a dict and return in a list seperated by a space.
    """
    stopwords_english = stopwords.words('english') # english stopwords
    lemmatizer = WordNetLemmatizer() #lemmatizer
    
    tokens = word_tokenize(value.lower()) #convert to lower case and tokenize
    clean_desc = [] #clean description
    for word in tokens:
        if (word not in stopwords_english #remove stopwords
            and word not in string.punctuation):
            lemmatize_word = lemmatizer.lemmatize(word) #lemmatize
            clean_desc.append(word) #convert 
    
    return ' '.join(clean_desc)


def process_features(dataframe):
    """
    Take as input d dataframe and remove redundant and unnecessary columns.
    Apply the following operations:
        1. Add a column released to capture in years the time between current year and the movie released year.
        2. Remove text from runtime.
        3. For numeric features apply regex to remove unwanted parts of the strings and convert the feature to type float.
        4. Select categorical columns with missing values and create a seperate value "missing" to replace for NaNs 
        5. Aggregate all categorical/object columns and create a feature named Description and apply extract_features.
        6. Scale all numeric values using the mix max scaler to convert all on the same scale.
    Return the processed dataframe.
    """
    dataframe = dataframe[dataframe.columns[~dataframe.columns.isin(['Title'])]].copy()
       
    dataframe['Released'] = datetime.now().year - pd.to_datetime(dataframe['Released']).dt.year #get time elapsed between current year and year the movie was released to utilize the temporal components
    
    vars_to_drop = ['index', 'Year', 'Poster', 'Ratings', 'DVD', 'BoxOffice', 'Website', 'Response', 'totalSeasons', 'Type']
    dataframe.drop(vars_to_drop, axis=1, inplace=True)
 
    dataframe['Plot'] = dataframe['Plot'].apply(lambda x: re.sub('[0-9]','',x)) #remove numbers from the plot   
    dataframe['Runtime'] = dataframe['Runtime'].apply(lambda x: re.sub('[a-z]','',x)) #remove text from runtime and convert to numeric
    
    num_vars = ['Runtime', 'Metascore', 'imdbRating', 'imdbVotes']
    for num_var in num_vars:
        try:
            dataframe[num_var] = dataframe[num_var].astype('float')
        except:
            dataframe[num_var] = dataframe[num_var].apply(lambda x: re.sub('(,)', '', x)).astype('float')
    
        dataframe[num_var] = dataframe[num_var].fillna(0)
    
    def clean_writer(value):
        """Clean the writer column"""
        try:
            value = re.sub(' \((.*)\)', '', value)
        except:
            pass
        return value
    
    dataframe['Writer'] = dataframe['Writer'].apply(clean_writer) #remove unnecessary text from values such as repeated (screenplay)
    
    cat_vars = [var for var in dataframe.columns if dataframe[var].isna().sum() > 0 and dataframe[var].dtype=='O'] #select categorical variables with missing values
    
    #create categories for those missing values
    for var in cat_vars: 
        dataframe[var] = dataframe[var].fillna(var+'Missing')
        
    dataframe['Description'] = dataframe[[var for var in dataframe.columns if dataframe[var].dtype=='O']].agg(','.join, axis=1).apply(extract_features) #combine categorical variables into a single variable and apply process features
    
    
    #scale numeric columns
    scaler = MinMaxScaler()
    dataframe[num_vars] = scaler.fit_transform(dataframe[num_vars])
    
    
    return dataframe[num_vars+['Description']]

