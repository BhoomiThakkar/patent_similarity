import os
import pathlib
import pandas as pd
from utility import calculate_cosine
from gensim.models.doc2vec import Doc2Vec
from . import configs

'''
1. Using the vectors output from train_vectors, compute similarity scores by grant year and cpc-4 digit classes, based 
on the input lag provided 
2. For each year, 1 file generated per cpc4 digit code 
'''

def compute_scores(lag, STARTYEAR, ENDYEAR):
    end_year = ENDYEAR
    start_year = STARTYEAR + lag

    model_dir = configs.model_dir
    score_dir = configs.score_dir
    data_dir = configs.data_dir

    data = pd.read_csv(data_dir + 'CleanedTextData-CPC-Iter2.csv')
    data['patent_id'] = data['patent_id'].astype('str')
    only_ids = data[['patent_id']]
    only_ids['patent_id'] = only_ids['patent_id'].astype('str')

    metadata = pd.read_csv(data_dir + 'PatentMetaData-CPC.csv')
    metadata['patent_id'] = metadata['patent_id'].astype('str')
    metadata_1 = pd.merge(metadata, only_ids, on=['patent_id'])
    metadata_1['patent_id'] = metadata_1['patent_id'].astype('str')
    metadata_1['cpc_4'] = metadata_1['cpc_4'].astype('str')

    # Calculate Scores
    for year in range(start_year,end_year+1):

        score_dir_year=score_dir+str(year)+'/'
        p = pathlib.Path(score_dir_year)
        if not p.is_dir():
            p.mkdir(parents=True)

        current_data=metadata_1[metadata_1['grant_year']==year]
        currentmodel_dir=model_dir+str(year)+'/'
        currentmodel=Doc2Vec.load(currentmodel_dir+'d2v.model')

        if lag==0:
            lagged_year=year
            laggedmodel=currentmodel
            lagged_data=current_data
        else:
            lagged_year=(year-lag)
            lagged_data=metadata_1[metadata_1['grant_year']==lagged_year]
            laggedmodel_dir=model_dir+str(lagged_year)+'/'
            laggedmodel=Doc2Vec.load(laggedmodel_dir+'d2v.model')

        cpcs=list(set(current_data.cpc_4))

        print(year, lagged_year)
        print('Current Year Data:{}, Lagged Year Data:{}, Unique CPC Sections:{}'.format(len(current_data),
                                                                                         len(lagged_data),len(cpcs)))

        if len(cpcs) == 0:
            print('\tLen (CPCs) = 0', year, lagged_year)
            continue

        for cpc in cpcs:
            current_data_ipc=current_data[current_data['cpc_4']==cpc]
            ids1=current_data_ipc['patent_id'].tolist()

            if len(ids1)==0:
                print('\tLen (ids2) = 0', year, lagged_year, cpc)
                continue

            lagged_data_ipc=lagged_data[lagged_data['cpc_4']==cpc]
            ids2=lagged_data_ipc['patent_id'].tolist()

            if len(ids2)==0:
                # print('\tLen (ids2) = 0', year, lagged_year, cpc)
                continue

            filepath=score_dir_year+str(cpc)+'.csv'

            if os.path.exists(filepath):
                continue

            calculate_cosine.compute_cosine_matrix(currentmodel=currentmodel, ids1=ids1,
                                                           previousmodel=laggedmodel, ids2=ids2, filepath=filepath)



