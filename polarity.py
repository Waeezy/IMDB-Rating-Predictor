import json
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
nltk.download()  #this has to be downloaded once
from nltk.tokenize import word_tokenize
import re


def cleaning(movie_data):
    # pd.set_option('display.max_columns', 500)
    # print(movie_data.iloc[[2]])

    movie_data = movie_data[['Title','Year', 'Runtime', 'Plot'
                            , 'Ratings', 'imdbRating', 'Awards', 'imdbVotes', 'imdbID', 'BoxOffice']]
    movie_data = movie_data.dropna()
    movie_data = movie_data[movie_data['Title'] != 'N/A']
    movie_data = movie_data[movie_data['Plot'] != 'N/A']
    movie_data = movie_data[movie_data['imdbRating'] != 'N/A']
    return movie_data


#Taken from here: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
stop_words = set(stopwords.words('english'))
def remove_stop(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return(filtered_sentence)

def comput_polarity(movie_data):
    movie_data['Plot'] = movie_data['Plot'].str.lower()
    movie_data['Title'] = movie_data['Title'].str.lower()

    #remove stop words(the,and...)
    movie_data['Plot'] = movie_data['Plot'].apply(lambda x: remove_stop(x))
    
    #keep just words and remove weird chars
    # regex = re.compile('[^a-zA-Z]')
    # regex.sub('', 'ab3d*E')
    movie_data['Plot'] = movie_data['Plot'].apply(lambda x: ' '.join(word for word in x))
    
    #compute polarity
    movie_data['plot_polarity'] = movie_data['Plot'].apply(lambda x: TextBlob(x).sentiment.polarity)

    return movie_data

def plot_stuff_and_ttest(movie_data):

    # movie_data['imdbRating'] = movie_data['imdbRating'].astype(str).str[0:3].astype(float)
    # fit = stats.linregress(movie_data['plot_polarity'], movie_data['imdbRating'])
    # plt.xlabel('Plot Polarity')
    # plt.ylabel('IMDB Rating')
    # plt.plot(movie_data['plot_polarity'], movie_data['imdbRating'], 'b.', alpha=0.5)
    # plt.plot(movie_data['plot_polarity'], movie_data['plot_polarity']*fit.slope + fit.intercept, 'r-', linewidth=2)
    # print(fit.slope)
    # print(fit.pvalue)
    # plt.show()

    # print("------------------------------")

    # # # clean the imbd votes
    movie_data['imdbVotes'] = movie_data['imdbVotes'].map(lambda x: x.replace(',', ''))
    movie_data = movie_data[ movie_data['imdbVotes'] != 'N/A' ]
    movie_data['imdbVotes'] = movie_data['imdbVotes'].astype('int32')
    movie_data = movie_data[ movie_data['imdbVotes'] < 60000 ]
    
    ax = plt.gca()      #To prevent plot from showing weird 1e6 notation for the y-axis
    ax.ticklabel_format(useOffset=False, style='plain')
    
    plt.xlabel('Plot Polarity')
    plt.ylabel('IMDB Votes')
    plt.plot(movie_data['plot_polarity'], movie_data['imdbVotes'], 'b.', alpha=0.5)
    plt.show()
    


def main(in_directory):
    movie_data = pd.read_json(in_directory, lines=True)

    movie_data = cleaning(movie_data)

    comput_polarity(movie_data)

    plot_stuff_and_ttest(movie_data)

    

 

if __name__ == "__main__":
    in_directory = './omdb-data.json.gz'
    # in_directory = sys.argv[1]
    # out_directory = 'sys.argv[2]
    main(in_directory)
