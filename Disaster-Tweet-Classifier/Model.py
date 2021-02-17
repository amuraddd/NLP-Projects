"""
Model
"""
import pandas as pd
import numpy as np
import Config
import joblib
from LoadData import read_data
from FeatureEngineering import pre_process_engineer
from FeatureSelection import scale_select
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

data = read_data(Config.PATH_TO_DATA, Config.SQL_QUERY, Config.DATA_INDEX)

def predict(dataframe):
    x_train, x_test, y_train, y_test = scale_select(dataframe)

    #load models
    logit = joblib.load(Config.LOGIT_MODEL)
    rf = joblib.load(Config.RANDOM_FOREST_MODEL)
    lasso = joblib.load(Config.LASSO_MODEL)

    #predict from models
    logit_predict_probs = logit.predict_proba(x_test)[:,1]
    rf_predict_probs = rf.predict_proba(x_test)[:,1]
    lasso_predict_probs = lasso.predict(x_test)

    #score the models
    logit_score = roc_auc_score(y_test, logit_predict_probs)
    rf_score = roc_auc_score(y_test, rf_predict_probs)
    lasso_score = roc_auc_score(y_test, lasso_predict_probs)
        
    #print scores
    print(100*("*"))
    
    print('ROC AUC Score from Logistic Classification: ', round(logit_score*100,2), '%')
    # Logistic Classifier Confusion Matrix
    logit_predict_binary = np.array([1 if y>0.4 else 0 for y in logit_predict_probs])
    logit_confusion_matrix = pd.crosstab(y_test.values.reshape(-1), 
                                         logit_predict_binary.reshape(-1), 
                                         rownames=['Actual'], 
                                         colnames=['Predicted'], 
                                         margins=True)
    print('\nLogit Regression Confusion Matrix:\n', logit_confusion_matrix)
    
    #roc curve for Logistic Classifier    
    ns_probs = [0 for _ in range(len(y_test))]
    
    ns_fpr, ns_tpr, _ = roc_curve(y_test.values, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test.values, logit_predict_probs)
    
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.title('Logistic Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # legend
    plt.legend()
    # plot
    plt.show()    
    
    print(100*("*"))
    
    print('ROC AUC Score from Random Forest Classification: ', round(rf_score*100,2), '%')
    # Random Forest Classifier Confusion Matrix
    rf_predict_binary = np.array([1 if y>0.4 else 0 for y in rf_predict_probs])
    rf_confusion_matrix = pd.crosstab(y_test.values.reshape(-1), 
                                         rf_predict_binary.reshape(-1), 
                                         rownames=['Actual'], 
                                         colnames=['Predicted'], 
                                         margins=True)
    #roc curve for Random Forest    
    ns_probs = [0 for _ in range(len(y_test))]
    
    ns_fpr, ns_tpr, _ = roc_curve(y_test.values, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test.values, rf_predict_probs)
    
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest Classifier')
    # axis labels
    plt.title('Random Forest')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # legend
    plt.legend()
    # plot
    plt.show()
    print('\nRandom Forest Confusion Matrix:\n', rf_confusion_matrix)
    
    #roc curve for Lasso Classifier    
    ns_probs = [0 for _ in range(len(y_test))]
    
    ns_fpr, ns_tpr, _ = roc_curve(y_test.values, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test.values, lasso_predict_probs)
    
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Lasso Classifier')
    # axis labels
    plt.title('Lasso')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # legend
    plt.legend()
    # plot
    plt.show()
    
    print(100*("*"))
    
    print('ROC AUC Score from Lasso Classification: ', round(lasso_score*100,2), '%')
    #Lasso Classifier Confusion Matrix
    lasso_predict_binary = np.array([1 if y>0.4 else 0 for y in lasso_predict_probs])
    lasso_confusion_matrix = pd.crosstab(y_test.values.reshape(-1), 
                                         lasso_predict_binary.reshape(-1), 
                                         rownames=['Actual'], 
                                         colnames=['Predicted'], 
                                         margins=True)
    print('\nLasso Classifier Confusion Matrix:\n', lasso_confusion_matrix)
    
    print(100*("*"))
    

#=================================Test models====================================================================
#model performance
predict(data)  



