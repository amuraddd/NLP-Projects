"""
Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import Config
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize']=12,5

def EDA(data):
    """
    Plot class distribution of the target variable.
    Plot bar chart of a count of missing values by columns with missing values greater than 0.
    Save the plot.
    """
    #pie chart for the target variable
    fig, ax = plt.subplots(figsize=(12,12))

    labels = config.PIE_CHART_LABELS
    target_values = data[config.TARGET_VALUE].value_counts()
    explode = [0.0, 0.1]

    patches, texts, autotexts = ax.pie(target_values, labels=labels, explode=explode,
                                       shadow=True, startangle=90, autopct='%1.1f%%',
                                      colors=config.PIE_CHART_COLORS)

    legend = ax.legend(patches,
                      target_values,
                      title=config.PIE_CHART_TITLE,
                      fontsize=20,
                      loc=config.PIE_CHART_LEGEND_LOC)

    plt.setp([legend.get_title(), autotexts, texts], size=18, weight="bold")
    ax.axis('equal') #ensure that the chart is drawen as a circle
    plt.style.use(config.PIE_CHART_STYLE)
    plt.show()
    fig.savefig(config.PIE_CHART_TITLE+'.png')
    plt.close(fig)

    #Columns with missing values
    cols = [col for col in data.columns if data[col].isna().sum()>0] #get columns with missing values
    missing_values = [data[col].isna().sum() for col in cols if data[col].isna().sum()>0] #get missing values for columns above

    fig, ax = plt.subplots(figsize=(12,7))

    x = np.arange(len(cols))
    width = 0.4

    rects = ax.bar(cols, missing_values, width, label=config.HISTOGRAM_LABELS,  color=config.HISTOGRAM_COLORS)
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=18)

    plt.xlabel(config.HISTOGRAM_XLABEL, fontsize=15)
    plt.ylabel(config.HISTOGRAM_YLABEL, fontsize=15)
    plt.xticks(cols, fontsize=15)
    plt.yticks(fontsize=15)
    fig.tight_layout()
    plt.show()
    fig.savefig(config.HISTOGRAM_XLABEL+'.png')
    plt.close(fig)
