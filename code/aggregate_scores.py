import os
import pathlib
import pandas as pd
from . import configs

'''
1. Collates the patent1, patent2 stats to generate mean, total similarity measures
2. One file generated for 1 set of scores 
'''


# Generate Aggregate Stats from Raw Patent1, Patent2 Scores for each class & year
def concatenate_year_wise(lag, STARTYEAR, ENDYEAR):
    end_year = ENDYEAR
    start_year = STARTYEAR + lag

    score_dir = configs.score_dir
    destination_dir = configs.compiled_scores
    p = pathlib.Path(destination_dir)
    if not p.is_dir():
        p.mkdir(parents=True)

    concatenated_df=pd.DataFrame()

    for year in range(start_year, end_year+1):
        score_dir_2 = score_dir + str(year) + '/'
        cumdf = pd.DataFrame()

        files = os.listdir(score_dir_2)
        print(year, len(files))

        ct=0

        for file in files:
            class1 = str(file)[:-4]

            tempdf = pd.read_csv(score_dir_2 + file)

            if len(tempdf) == 0:
                continue

            columns1 = tempdf.columns

            allowed = []
            for c in tempdf.columns:
                if 'Unnamed' not in c:
                    allowed.append(c)

            ct += 1
            if ct%100 == 0 and ct != 0:
                print('\tCheckpoint: {}/{}'.format(ct,len(files)))

            tempdf['mean_similarity_'+str(lag)] = tempdf[allowed].mean(axis=1)
            tempdf['median_similarity_'+str(lag)] = tempdf[allowed].median(axis=1)
            tempdf['total_similarity_'+str(lag)] = tempdf[allowed].sum(axis=1)
            tempdf['peer_count']=len(columns1)-1
            tempdf['skewness_'+str(lag)] = tempdf[allowed].skew(axis=1)
            tempdf['kurtosis_'+str(lag)] = tempdf[allowed].kurt(axis=1)
            tempdf['std_dev_similarity_'+str(lag)] = tempdf[allowed].std(axis=1)

            tempdf.rename(columns={'Unnamed: 0': 'patent_id'}, inplace=True)
            tempdf1 = tempdf[['patent_id', 'mean_similarity_'+str(lag),
                              'median_similarity_'+str(lag), 'total_similarity_'+str(lag),
                              'peer_count','skewness_'+str(lag),'kurtosis_'+str(lag),
                              'std_dev_similarity_'+str(lag)]]

            cumdf = pd.concat([cumdf, tempdf1])

        print(year, len(cumdf))
        print('Writing file to destination .. ')
        concatenated_df=pd.concat([cumdf,concatenated_df])
        cumdf.to_csv(destination_dir + str(year) + '.csv', index=False)

    return concatenated_df


# Collate all class files into a single file ..
def concatenate_score_set(lag, STARTYEAR, ENDYEAR):
    end_year = ENDYEAR
    start_year = STARTYEAR + lag

    concatenated_score_dir= configs.compiled_scores_year
    destination_dir= configs.compiled_scores

    concatenated_df=pd.DataFrame()
    for year in range(start_year, end_year+1):
        tempdf1=pd.read_csv(concatenated_score_dir+str(year)+'.csv',
                            usecols=['patent_id', 'mean_similarity_'+str(lag),
                              'median_similarity_'+str(lag), 'total_similarity_'+str(lag),
                              'peer_count','skewness_'+str(lag),'kurtosis_'+str(lag),
                              'std_dev_similarity_'+str(lag)])

        concatenated_df = pd.concat([concatenated_df, tempdf1])
        print(year, len(tempdf1))

    concatenated_df=concatenated_df['patent_id', 'mean_similarity_'+str(lag),
                              'median_similarity_'+str(lag), 'total_similarity_'+str(lag),
                              'peer_count','skewness_'+str(lag),'kurtosis_'+str(lag),
                              'std_dev_similarity_'+str(lag)]

    print('File written to ' + destination_dir)
    concatenated_df.to_csv(destination_dir+'Lagged_'+str(lag)+'.csv',index=False)

    return concatenated_df


