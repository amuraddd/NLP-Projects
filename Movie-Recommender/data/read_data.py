"""
Read Data
"""
import sqlite3
import pandas as pd

def read_data(db_name, query_file):
    """
    Read the table from the SQLite database which contains data on movies retreived from IMDB.
    """
    con = sqlite3.connect(db_name)
    cursor = con.cursor()

    sql = open(query_file,'r')
    query = sql.read()
    sql.close()

    data = pd.read_sql_query(query, con=con)
    data.drop_duplicates(subset=['Title'], inplace=True)
    data = data[data['Type']=='movie']
    data.set_index('imdbID', inplace=True)

    con.commit()
    con.close()

    return data
