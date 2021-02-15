"""
Feature Engineering and Selection
"""
import numpy as np
import pandas as pd
import joblib
import Config
import nltk
import string
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from PreProcessing import clean_text, replace_missing, create_rare_label

#===process tweet texts
def tweet_preprocess(text):
    """
    Imput tweet text.
    Remove stopwords and punctuation. Tokenize and Lemmatize the text.
    Return cleaned tweet text.
    Get applied inside map_probs_process_tweets.
    """
    stopwords_english = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text.lower())
    clean_desc = []
    for word in tokens:
        if (word not in stopwords_english
           and word not in string.punctuation):
            lemmatized_word = lemmatizer.lemmatize(word)
            clean_desc.append(lemmatized_word)
    return ' '.join(clean_desc)

#===Create probabilities for categorical variables
def map_probs_process_tweets(dataframe, cat_cols = Config.CAT_COLS):
    """
    Map the labels to their respective probabilities calculated as proportions.
    Return the dataframe.
    """
    for col in cat_cols:
        probs = dataframe[col].value_counts()/dataframe[col].value_counts().sum()
        dataframe[col] = dataframe[col].map(probs)
    dataframe[Config.TWEET_TEXT] = dataframe[Config.TWEET_TEXT].apply(tweet_preprocess)

    return dataframe

#=======Vectorize tweets
def vectorize_tweets(dataframe, vectorize_column=Config.TWEET_TEXT, vectorizer=Config.VECTORIZER):
    vectorizer = joblib.load(vectorizer)
    vectorizer_transform = vectorizer.transform(dataframe[vectorize_column])
    dataframe = pd.DataFrame(data=np.hstack([dataframe[Config.COLS_TO_COMBINE],
                                vectorizer_transform.toarray()]),
                                index=dataframe.index,
                                columns=Config.COLS_TO_COMBINE+vectorizer.get_feature_names())
    return dataframe

#=====Apply feature engineering and pre-processing
def pre_process_engineer(dataframe):
    dataframe = create_rare_label(replace_missing(clean_text(dataframe)))
    dataframe = vectorize_tweets(map_probs_process_tweets(dataframe))
    return dataframe
