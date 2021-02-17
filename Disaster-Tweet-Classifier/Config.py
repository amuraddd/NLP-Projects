#==========PATHS================================================================================
PATH_TO_DATA = 'data/Disaster_Tweets.db'
VECTORIZER = 'models/vectorizer.pkl'
SQL_QUERY = 'data/testing_query.sql'
DATA_INDEX = ['id']

#==========GLOBAL===============================================================================
TARGET_VALUE = 'target'
TWEET_TEXT = 'text'

#============EDA PARAMETERS=====================================================================
#PIE CHART
PIE_CHART_LABELS = ['Disaster', 'No Disaster']
PIE_CHART_COLORS = ['#cd5d7d', '#9ad3bc']
PIE_CHART_LEGEND_LOC = 'upper left'
PIE_CHART_TITLE = 'Class Distribution'
PIE_CHART_STYLE = 'seaborn-dark'

#MISSING VALUES HISTOGRAM
HISTOGRAM_LABELS = 'Missing Values'
HISTOGRAM_COLORS = ['#6e5773', '#c06c84']
HISTOGRAM_XLABEL = 'Columns with Missing Values'
HISTOGRAM_YLABEL = 'Count'

#===========PRE-PROCESSING & FEATURE-ENGINEERING========================================================================
COLS_TO_CLEAN = ['keyword', 'location', 'text']
KEYWORD_REGEX = '%|\d+'
LOCATION_REGEX = '<|>|=|;|:|\'|!|@|\/|#|$|%|\^|&|\*|\[|\]|\.|-|_|\d+|\+|\?|Û|¢|\(|\)|\||\s+|,'
TEXT_REGEX = 'https|<|>|=|;|:|\'|!|@|\/|#|$|%|\^|&|\*|\[|\]|\.|-|_|\d+|\+|\?|Û|¢|\(|\)|\||,'

CAT_COLS = ['keyword', 'location']
RARE_PCT = 0.001

#TFIDF Vectorizer
COLS_TO_COMBINE = ['keyword', 'location']

#============Feature Selection=======================================================================
SELECTED_FEATURES = 'data/selected_features.csv'


#============Models======================================================================================================
SCALER = 'models/scaler.pkl'
LOGIT_MODEL = 'models/logistic_model.pkl'
RANDOM_FOREST_MODEL = 'models/random_forest_model.pkl'
LASSO_MODEL = 'models/lasso_model.pkl'
