import pandas as pd
from . import configs

'''
1. Collate Patent Grant Information from patent.tsv, Application Information from application.tsv and 
CPC class information from cpc_current.tsv
2. Output generated is PatentMetaData-CPC.csv and OnlyPatentText-CPC.csv 
'''

project_dir= configs.project_dir
source_dir= configs.source_dir
destination_dir= configs.data_dir
stats_destination_dir= configs.stats_dir

patents=pd.read_csv(source_dir+'patent.tsv', sep='\t')
patents.rename(columns={'date':'grant_date'},inplace=True)
patents.rename(columns={'id':'patent_id'},inplace=True)
patents.rename(columns={'number':'patent_number'},inplace=True)
patents['patent_id']=patents['patent_id'].astype('str')

applications=pd.read_csv(source_dir+'application.tsv', sep='\t')
applications.rename(columns={'date':'filing_date'},inplace=True)
applications.rename(columns={'id':'application_id'},inplace=True)
applications.rename(columns={'number':'application_number'},inplace=True)
applications['patent_id']=applications['patent_id'].astype('str')


patent_data=pd.merge(patents,applications,on=['patent_id'])  # 7,526,704
patent_data.drop(['country_y'],inplace=True,axis=1)
patent_data.rename(columns={'country_x':'country'},inplace=True)


cpc_current=pd.read_csv(source_dir+'cpc_current.tsv',sep='\t')  # 41413742
cpc_current_1=cpc_current[cpc_current['sequence']==0]   # 6,779,677
cpc_current_1['patent_id']=cpc_current_1['patent_id'].astype('str')
patent_data_class=pd.merge(patents,cpc_current_1,on=['patent_id'])  # 6779677
patent_data_class_1=patent_data_class.drop_duplicates(subset=['patent_id']) # 6779677
patent_data_class_1['cpc_4']=patent_data_class_1['group_id'].str[:4]

patent_data_class_2=pd.merge(patent_data_class_1,applications,on=['patent_id'])  # 6779677
patent_data_class_3=patent_data_class_2.drop_duplicates(subset=['patent_id']) # 6779677
patent_data_class_3.drop(['country_y'],axis=1,inplace=True)
patent_data_class_3.rename(columns={'country_x':'country'},inplace=True)

patent_data_class_3['grant_year']=patent_data_class_3['grant_date'].str[:4]
patent_data_class_3['grant_year']=patent_data_class_3['grant_year'].astype('float64').astype('int64')

patent_data_class_3['filing_year']=patent_data_class_3['filing_date'].str[:4]
patent_data_class_3['grant_year']=patent_data_class_3['grant_year'].astype('float64').astype('int64')

only_text=patent_data_class_3[['patent_id','grant_year','filing_year','title','abstract']]
only_text.to_csv(destination_dir+'OnlyPatentText-CPC.csv',index=False)
meta_data=patent_data_class_3.drop(['title','abstract'],axis=1)
meta_data.to_csv(destination_dir+'PatentMetaData-CPC.csv',index=False)

df3=meta_data.groupby(by=['filing_year']).count()
df3=df3[['patent_id']]
df3=df3.reset_index()
df3.to_csv(stats_destination_dir+'PatentCountLog-FilingYear.csv',index=False)

df3=meta_data.groupby(by=['grant_year']).count()
df3=df3[['patent_id']]
df3=df3.reset_index()
df3.to_csv(stats_destination_dir+'PatentCountLog-GrantYear.csv',index=False)

