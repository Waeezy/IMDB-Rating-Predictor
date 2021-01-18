import json
import re
import sys

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LinearRegression
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from scipy import stats
# from textblob import TextBlob
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import FunctionTransformer, StandardScaler
# from sklearn.svm import SVC
# nltk.download()  #this has to be downloaded once

# stop_words = set(stopwords.words('english'))
# def remove_stop(text):
#     word_tokens = word_tokenize(text)
#     filtered_sentence = [w for w in word_tokens if not w in stop_words]
#     filtered_sentence = []
#     for w in word_tokens:
#         if w not in stop_words:
#             filtered_sentence.append(w)
#     return(filtered_sentence)

pd.options.mode.chained_assignment = None  # default='warn'

def map_dict(words):
    d = dict()
    i = 0
    for word in words:
        if word not in d:
            d[word] = i
            i += 1
    return d

def clean_data(df):
    df = df[[ 
        'Year', 
        'Rated', 
        'Genre', 
        # 'Director', 
        # 'Writer', 
        # 'Actors', 
        'Language',
        'Country', 
        'imdbRating', 
        # 'Production',
        'imdbVotes',
        'Runtime',
        # 'Plot',
        # 'Title'
    ]]

    # clean the imdb rating
    df = df[ df['imdbRating'] != 'N/A' ]
    df['imdbRating'] = df['imdbRating'].astype('float32')*10
    df['imdbRating'] = df['imdbRating'].astype('int32')
    df = df[df['imdbRating'] >= 0] 
    df = df[df['imdbRating'] <= 100]

    # clean the imbd votes
    df['imdbVotes'] = df['imdbVotes'].map(lambda x: x.replace(',', ''))
    df = df[ df['imdbVotes'] != 'N/A' ]
    df['imdbVotes'] = df['imdbVotes'].astype('int32')

    # clean the year
    df['Year'] = df['Year'].astype(str).str[0:4].astype(int)

    # clean the Genre 
    df['Genre'] = df['Genre'].str.split(", ")
    df['Genre1'] = df['Genre'].map(lambda x: x[0])
    df['Genre2'] = df['Genre'].map(lambda x: x[1] if len(x) > 1 else 'N/A')
    df['Genre3'] = df['Genre'].map(lambda x: x[2] if len(x) > 2 else 'N/A')
    df = df.drop('Genre', axis=1)
    df = df.replace({"Genre1": map_dict(df['Genre1'].unique())})
    df = df.replace({"Genre2": map_dict(df['Genre2'].unique())})
    df = df.replace({"Genre3": map_dict(df['Genre3'].unique())})

    # clean the rated
    df['Rated'] = df['Rated'].replace(['Not Rated', 'NOT RATED', 'Unrated', 'UNRATED'], 'N/A')
    df = df.replace({"Rated": map_dict(df['Rated'].unique())})

    # clean the language
    df['Language'] = df['Language'].str.split(", ")
    df['Language1'] = df['Language'].map(lambda x: x[0])
    df['Language2'] = df['Language'].map(lambda x: x[1] if len(x) > 1 else 'N/A')
    df = df.drop('Language', axis=1)
    df = df.replace({"Language1": map_dict(df['Language1'].unique())})
    df = df.replace({"Language2": map_dict(df['Language2'].unique())})

    # clean the country
    df['Country'] = df['Country'].str.split(", ")
    df['Country1'] = df['Country'].map(lambda x: x[0])
    df['Country2'] = df['Country'].map(lambda x: x[1] if len(x) > 1 else 'N/A')
    df = df.drop('Country', axis=1)
    df = df.replace({"Country1": map_dict(df['Country1'].unique())})
    df = df.replace({"Country2": map_dict(df['Country2'].unique())})

    # clean runtime
    df = df[ df['Runtime'] != 'N/A']
    df['Runtime'] = df['Runtime'].str.extract('(\d+)')
    df['Runtime'] = df['Runtime'].astype('int32')

    # clean Director
    # df = df.replace({"Director": map_dict(df['Director'].unique())})
    
    # clean actor
    # df['Actors'] = df['Actors'].str.split(", ")
    # df['Actors1'] = df['Actors'].map(lambda x: x[0])
    # df['Actors2'] = df['Actors'].map(lambda x: x[1] if len(x) > 1 else 'N/A')
    # df['Actors3'] = df['Actors'].map(lambda x: x[2] if len(x) > 2 else 'N/A')
    # df = df.drop('Actors', axis=1)
    # df = df.replace({"Actors1": map_dict(df['Actors1'].unique())})
    # df = df.replace({"Actors2": map_dict(df['Actors2'].unique())})
    # df = df.replace({"Actors3": map_dict(df['Actors3'].unique())})
    
    # calculate poloarity
    # df['Plot'] = df['Plot'].str.lower()
    # df['Title'] = df['Title'].str.lower()
    # df['Plot'] = df['Plot'].apply(lambda x: remove_stop(x))
    # df['Title'] = df['Title'].apply(lambda x: remove_stop(x))
    # df['Plot'] = df['Plot'].apply(lambda x: ' '.join(word for word in x))
    # df['Title'] = df['Title'].apply(lambda x: ' '.join(word for word in x))
    # df['plot_polarity'] = df['Plot'].apply(lambda x: TextBlob(x).sentiment.polarity)
    # df['title_polarity'] = df['Plot'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    return df

def get_XY(df):
    # Choosing the input features to create the best model
    X = df[[
        'Year',
        'Genre1',
        'Genre2',
        'Genre3',
        'Rated',
        'Language1',
        'Language2',
        'Country1',
        'Country2',
        'Runtime',
        # 'Actors1',
        # 'Actors2', 
        # 'Actors3',
        # 'Director',
        # 'plot_polarity',
        # 'title_polarity'
        # 'imdbVotes',
    ]].to_numpy()
    y = df['imdbRating']
    return X, y

def run_model(df):
    # get X,y and split data sets
    X, y = get_XY(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # fit and test the model
    model = RandomForestClassifier(n_estimators=50, max_depth=10)
    model.fit(X_train, y_train)
    predicted = np.array(model.predict(X_test))
    y_test = np.array(y_test)
    score = np.average(np.absolute(predicted - y_test) / y_test)*100
    return score

def main(in_directory):
    df = pd.read_json(in_directory, lines=True)

    # clean dataset
    df = clean_data(df)

    print("Training the model 100 times and averaging out the errors:\n")
    results = []
    for i in range(100):
        error = run_model(df)
        results.append(error)
        print("Model {} Error: {:.2f}%".format(i, error))
    print("\nOverall Average Error: {:.2f}%".format(float(sum(results)/len(results))))
    
if __name__ == "__main__":
    in_directory = sys.argv[1]
    main(in_directory)
