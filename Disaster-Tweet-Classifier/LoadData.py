"""
READ DATA
"""
import sqlite3
import pandas as pd

def read_data(db_path, query_path, index):
    """
    Read data from the SQLite Database
    Function Parameters:
    db_path: database path
    query_path: sql query path
    index: index of the dataframe after the data is read from database
    """
    db = sqlite3.connect(db_path)
    cursor = db.cursor()

    sql = open(query_path, 'r')
    query = sql.read()
    sql.close()

    data = pd.read_sql_query(query, con=db)
    data.set_index(index, inplace=True)

    db.commit()
    db.close()

    return data
