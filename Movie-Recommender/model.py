"""
Model
"""
import glob
import pandas as pd
import numpy as np
from data.read_data import read_data
from feature_selection import feature_selection

"""Read the data"""
db_name = glob.glob('data\*.db')[0]
query_file = glob.glob('data\*.sql')[0]
main = read_data(db_name, query_file)

def similarity(dataframe):
    """
    Take in as input a dataframe. Apply feature selection.
    Keep 10% of the data for randomly selecting a movie and call this test.
    Keep 90% of the data for computing cosine similarity for the selected movie.
    Compute consine similarity between the selected movie and the rest of the data froom the above step.
    Sort the movies in descending order and get the 5 most similar movies.
    Return the test movie and the similar movies.
    """
    main = dataframe
    
    dataframe = feature_selection(dataframe)
    train_size = round((len(dataframe)*0.9))
    train = dataframe[:train_size]
    test = dataframe[train_size:]
    
    test_value = test.iloc[np.random.randint(0,10),:]
    
    #compute cosine similarity
    neighbors = {}
    for i, r in train.iterrows():
        similarity = np.dot(test_value,r)/(np.linalg.norm(test_value)*np.linalg.norm(r))
        neighbors[i] = similarity
    
    #get similary movies in descending order
    neighbors = {k: v for k, v in sorted(neighbors.items(), key=lambda item: item[1], reverse=True)}
    
    test_final = pd.concat([test, main], axis=1, sort=False)
    train_final = pd.concat([train, main], axis=1, sort=False)
    
    test_movie = test_final.loc[test_value.name,['Title', 'Rated', 'Genre', 'imdbRating']]
    similar_movies = train_final.loc[list(neighbors.keys())[:5],['Title','Rated', 'Genre', 'Released', 'imdbRating']]
    
    return test_movie, similar_movies

selected_movie, similar_movies = similarity(main)

print(100*"*")
print("Since you like: \n", selected_movie)
print(100*"*")
print("We think you might as also like: \n", similar_movies)
print(100*"*")