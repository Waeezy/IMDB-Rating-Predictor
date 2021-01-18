import requests
import json, sys
import pandas as pd
from pathlib import Path
import re
from json.decoder import JSONDecodeError


# API_KEY = sys.argv[1]
# API_KEY = "e826429c"     #Limited
API_KEY = "77ba394b"
cachefile = 'omdb-cache.dbm'
NOT_FOUND = 'notfound'
request_limit = False
OUTPUT_FILE_NAME = './omdb-data.json.gz'

def get_omdb_data(imdb_id):
    
    global request_limit
    if request_limit:
        return None
    if not imdb_id.startswith('tt'):
        raise ValueError('movies only')

    url = 'http://www.omdbapi.com/?i=%s&apikey=%s&plot=full' % (imdb_id, API_KEY)
    print('fetching', url)
    r = requests.get(url)

    r = re.sub(r'^jsonp\d+\(|\)\s+$', '', r.text)
    data = json.loads(r)

    try:
        if data['Response'] == 'False':
            if data['Error'] == 'Error getting data.':
                print("Error getting data")
                return NOT_FOUND
            elif data['Error'] == 'Request limit reached!':
                print("Request limit reached")
                request_limit = True
                return None
            else:
                raise ValueError(data['Error'])
    except JSONDecodeError as e:
        print ('Decoding JSON has failed')

    return data

def get_movies_and_save_them():
    # get movies ID's from file
    #infile = './moviequery.json.gz'
    infile = './wikidata-movies.json.gz'
    movie_data = pd.read_json(infile, orient='records', lines=True)
    movie_data = movie_data[:10000]  #TODO: Get more rows later

    # get movies data from OMDB using ID's
    #test_db = movie_data['IMDB_ID'].apply(get_omdb_data)
    test_db = movie_data['imdb_id'].apply(get_omdb_data)
    test_db = test_db[test_db.notnull()]
    # print(test_db)
    
    # save movies data into file
    test_db.to_json(OUTPUT_FILE_NAME, orient='records', lines=True, compression='gzip')

def cleaning(movie_data):
    # pd.set_option('display.max_columns', 500)
    # print(movie_data.iloc[[2]])

    movie_data = movie_data[['Title','Year', 'Runtime', 'Genre', 'Director', 'Writer', 'Actors'
                            , 'Ratings', 'imdbRating', 'Awards', 'imdbVotes', 'imdbID', 'BoxOffice']]
    movie_data = movie_data.dropna()
    # movie_data = movie_data[movie_data['BoxOffice'] != 'N/A']
    movie_data = movie_data[movie_data['Title'] != 'N/A']
    movie_data = movie_data[movie_data['Writer'] != 'N/A']
    movie_data = movie_data[movie_data['Actors'] != 'N/A']
    movie_data = movie_data[movie_data['imdbRating'] != 'N/A']
    return movie_data

def main(): 
    if not Path(OUTPUT_FILE_NAME).is_file(): get_movies_and_save_them()
    movie_data = pd.read_json(OUTPUT_FILE_NAME, orient='records', lines=True, compression='gzip')
    movie_data = cleaning(movie_data)
    print(movie_data)
    


if __name__ == '__main__':
    main()
