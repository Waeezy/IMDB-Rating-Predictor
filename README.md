# CMPT353-Final-Project
## An Exploration of Movie Data Statistics

To run the programs, please install the following:
```
$ pip install -U textblob
$ python -m textblob.download_corpora
$ pip install -U matplotlib stats sklearn stopwords
```
This will install textblob, which is used to answer questions 2 and 3.

### get_data.py
This program will use the imdb id provided to fetch data from OMDb and then stores it in the _omdb-data.json.gz_ file.
Run this script like so:
```
$ python get_data.py omdb-data.json.gz
```

### rating_compare.py
This program produces an output file with the results of various tests. It also creates a single histogram for the untransformed data.
Please uncomment the `plt.show()` line to view the data. 
Run this script like so:
```
$ python rating_compare.py
```

### polarity.py
This program produce the polarity for the plots and it creates two graphs (plot polarity vs IMDB ratings) and (plot polarity vs IMDB votes).
Please uncomment the `plt.show()` line to view the data. 
when running the script for the first time nltk downloader will ask to download some packeges.
Run this script like so:
```
$ python polarity.py
```

### rating_prediction.py
This program identify a combination of the movie features that can best predict its rating.
Run this script like so:
```
$ python rating_prediction.py .\omdb-data.json.gz
```