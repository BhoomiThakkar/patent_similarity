import pathlib
import pandas as pd
from utility import train_model
from . import configs

'''
1. Use the cleaned textual data to train vector for each patent by grant year
2. Training is divided by year for ease of computation
3. Vector size = 100, Learning rate (alpha) = 0.05, Distributed Mean variant for Doc2Vec 
4. One model saved for each year
'''

def train_doc2vec(alpha, STARTYEAR, ENDYEAR):
    model_dir= configs.model_dir
    data=pd.read_csv(configs.data_dir + 'CleanedTextData-CPC-Iter2.csv')
    data['patent_id']=data['patent_id'].astype('str')

    only_ids=data[['patent_id']]
    only_ids['patent_id']=only_ids['patent_id'].astype('str')

    metadata=pd.read_csv(configs.data_dir + 'PatentMetaData-CPC.csv')
    metadata['patent_id']=metadata['patent_id'].astype('str')
    metadata_1=pd.merge(metadata,data,on=['patent_id'])

    # Vector Training
    for year in range(STARTYEAR,ENDYEAR+1):
        data_1=metadata_1[metadata_1['grant_year']==year]
        model_dir_year = model_dir + str(year) + '/'
        p = pathlib.Path(model_dir_year)
        if not p.is_dir():
            p.mkdir(parents=True)

        print(year,len(data_1))

        train_model.train_model(data=data_1['cleaned_text'].tolist(), ids=data_1['patent_id'].tolist(),
                                destination_dir=model_dir_year, alpha=alpha, type1='')


