"""
Feature Selection
"""
import joblib
import Config
import pandas as pd
from FeatureEngineering import pre_process_engineer
from sklearn.model_selection import train_test_split


def scale_select(dataframe):
    """
    Take as input the priginal data. Pre-process and engineer features. Scale all features.
    Return x and y variables for modeling.
    """
    dataframe = dataframe.copy()
    x = pre_process_engineer(dataframe)

    #scale data
    scaler = joblib.load(Config.SCALER)
    scaled_x = scaler.transform(x)
    scaled_data = pd.DataFrame(scaled_x, index=dataframe.index, columns=x.columns)
    y = dataframe[[Config.TARGET_VALUE]]

    #load selected features
    features = pd.read_csv(Config.SELECTED_FEATURES, index_col=[0])
    features = features['0'].to_list()

    #train test split
    x_train, x_test, y_train, y_test = train_test_split(scaled_data,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=0)
    #define x_train and x_test by selecting features
    x_train = x_train[features]
    x_test = x_test[features]

    return x_train, x_test, y_train, y_test
