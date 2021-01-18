import json
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

def main(in_directory):
    df = pd.read_json(in_directory, lines=True)
    df = df[['Title', 'Year', 'Ratings', 'Metascore', 'imdbRating', 'imdbVotes']]
    df[['1','RottenToms', '3']] = pd.DataFrame(df['Ratings'].tolist(), index= df.index)
    df = df[['Title', 'Year', 'Ratings', 'Metascore', 'imdbRating', 'imdbVotes', 'RottenToms']]
    # df['RottenToms'] = df['RottenToms']['Value']

    # relationship between Year and IMdb rating
    df2 = df[['Year', 'imdbRating']]
    df2['Year'] = df2['Year'].astype(str).str[0:4].astype(int)
    df2['imdbRating'] = pd.to_numeric(df2['imdbRating'], errors='coerce').fillna(0).astype(np.float64)
    df2 = df2.groupby(['Year']).mean().reset_index()
    fit = stats.linregress(df2['Year'], df2['imdbRating'])

    # relationship between Year and MetaScore
    print(df['Ratings'])
    # print(df.count())

    # plt.xticks(rotation=100)
    plt.plot(df2['Year'], df2['imdbRating'], 'b.', alpha=0.5)
    plt.plot(df2['Year'], df2['Year']*fit.slope + fit.intercept, 'r-', linewidth=3)
    plt.show()
    print(fit.slope)

if __name__ == "__main__":
    in_directory = sys.argv[1]
    main(in_directory)
