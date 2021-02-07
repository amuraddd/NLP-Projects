"""
Feature Selection
"""
import pandas as pd
import numpy as np
from preprocessing import process_features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def feature_selection(dataframe):
    """
    Take as input a dataframe. Apply process features. Apply the following operations:
        1. Vectorize the description column to get term frequency inverse document frequecy.
        2. Create a dataframe without the description column but with all the feature names from TFIDF.
        3. TFIDF produces a sparse matrix with significantly more features than the instances in the dataframe.
        4. Apply PCA to reduce dimensionality of teh data and select the first 10 components from PCA as well as plot the variance explained per component.
    Return the processed dataframe.
    """
    dataframe = process_features(dataframe)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(dataframe['Description'])
    dataframe[vectorizer.get_feature_names()] = pd.DataFrame(vectorizer.transform(dataframe['Description']).toarray(), index=dataframe.index)
    dataframe = dataframe[dataframe.columns[~dataframe.columns.isin(['Description'])]]
    print('Data before dimension reduction: ', dataframe.shape)    
    
    pca = PCA(n_components=700, svd_solver='randomized')
    dataframe = pd.DataFrame(pca.fit_transform(dataframe), index=dataframe.index)
    
    print('Data after dimension reduction: ', dataframe.shape)  
    print('Variation explained: ', round(sum(pca.explained_variance_ratio_[:700])*100,2),'%')
    plt.plot(pca.explained_variance_ratio_[:700]*100)
    plt.ylabel('Variance Explained')
    plt.xlabel('Component')
    plt.show()
    
    return dataframe
