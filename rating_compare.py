import json
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {ttest_p:.3g}\n"
    "Original data normality p-values: {imdb_normality_pvalue:.3g} {metascore_normality_pvalue:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {t_imdb_normality_pvalue:.3g} {t_metascore_normality_pvalue:.3g}\n"
    "Transformed data equal-variance p-value: {t_levene_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_pvalue:.3g}"
)


def main(in_directory):
    df = pd.read_json(in_directory, lines=True)
    df = df[['Title', 'Metascore', 'imdbRating']]
    #df[['1','RottenToms', '3']] = pd.DataFrame(df['Ratings'].tolist(), index= df.index)
    
    #clean imdbrating data, remove all "N/A" values, convert to int, divide by 10
    imdb_rating = df[['Title', 'imdbRating']]
    imdb_rating = imdb_rating[imdb_rating.imdbRating != "N/A"]
    imdb_rating['imdbRating'] = imdb_rating['imdbRating'].astype(str).str[0:3].astype(float)
    imdb_rating = imdb_rating.reset_index(drop = True)
    #print(imdb_rating)
    
    #clean metascore data, remove all "N/A" values, convert to int, divide by 10
    metascore_rating = df[['Title','Metascore']]
    metascore_rating = metascore_rating[metascore_rating.Metascore != "N/A"]
    metascore_rating['Metascore'] = metascore_rating['Metascore'].astype(str).str[0:2].astype(int)
    metascore_rating['Metascore'] = metascore_rating['Metascore']/10
    metascore_rating = metascore_rating.reset_index(drop = True)
    #print(metascore_rating)
    
    #T-test
    #print(imdb_rating.imdbRating.mean()) # = 6.03~
    #print(metascore_rating.Metascore.mean()) # = 5.68~
    ttest_pvalue = stats.ttest_ind(imdb_rating['imdbRating'], metascore_rating['Metascore']).pvalue
    #print(ttest_pvalue)
    
    #U-test
    utest_pvalue = stats.mannwhitneyu(imdb_rating['imdbRating'], metascore_rating['Metascore']).pvalue
    
    #do some plots and see what happens
    #plt.hist(imdb_rating['imdbRating'], density = True, bins = 10, alpha = 0.5)
    #plt.hist(metascore_rating['Metascore'], density = True, bins = 10, alpha = 0.5)
    #plt.show()
        
    #find out if data is normally distributed
    imdb_normality_pvalue = stats.normaltest(imdb_rating['imdbRating']).pvalue
    metascore_normality_pvalue = stats.normaltest(metascore_rating['Metascore']).pvalue
    #if pvalue > 0.05 then data normally dist
    #print(imdb_normality_pvalue)
    #print(metascore_normality_pvalue) 
    #check for equal variance
    initial_levene = stats.levene(imdb_rating['imdbRating'], metascore_rating['Metascore']).pvalue
    
    #transform
    t_imdb = np.power(imdb_rating['imdbRating'],2)
    t_metascore = np.power(metascore_rating['Metascore'],2)
    
    t_imdb_normality_pvalue = stats.normaltest(t_imdb).pvalue
    t_metascore_normality_pvalue = stats.normaltest(t_metascore).pvalue
    
    #print(t_imdb_normality_pvalue)
    #print(t_metascore_normality_pvalue)
    t_levene_p = stats.levene(t_imdb, t_metascore).pvalue
    #print(t_levene_p)
    
    plt.hist(imdb_rating['imdbRating'], density = True, bins = 10, alpha = 0.5, label = 'IMDb')
    plt.hist(metascore_rating['Metascore'], density = True, bins = 10, alpha = 0.5, label = 'Metascore')
    plt.title('Histogram of Movie Ratings vs Frequency Density')
    plt.xlabel('Movie Ratings')
    plt.ylabel('Frequency Density')
    plt.legend()
    plt.show()
    
    f = open("comparision_output.txt", "a")
    f.write(OUTPUT_TEMPLATE.format(
        ttest_p = ttest_pvalue,
        imdb_normality_pvalue = imdb_normality_pvalue,
        metascore_normality_pvalue = metascore_normality_pvalue,
        initial_levene_p = initial_levene,
        t_imdb_normality_pvalue = t_imdb_normality_pvalue,
        t_metascore_normality_pvalue = t_metascore_normality_pvalue,
        t_levene_p = t_levene_p,
        utest_pvalue = utest_pvalue,
    ))
    f.close()
if __name__ == "__main__":
    in_directory = sys.argv[1]
    main(in_directory)