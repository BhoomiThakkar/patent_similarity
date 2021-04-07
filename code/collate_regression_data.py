import pandas as pd
from . import configs

'''
1. Compiles the scores with other variables like KPSS market values and citations 
'''

project_dir= configs.project_dir
data_dir= configs.data_dir
score_dir= configs.score_dir


def compile_regression_data(lag):
    concatenated_df=pd.read_csv(score_dir+'Lagged_'+str(lag)+'.csv')  # 6,708,963
    concatenated_df['application_novelty']=1-concatenated_df['mean_similarity_'+str(lag)]

    metadata=pd.read_csv(data_dir+'PatentMetaData-CPC.csv')  # 6,779,677
    forward_citations=pd.read_csv(data_dir+'total_forward_citations_dec20.csv')  # 7,583,358
    backward_citations=pd.read_csv(data_dir+'total_backward_citations_dec20.csv')  # 7,074,837
    kpss_data=pd.read_csv(data_dir+'KPSS_2019_public.csv') # 2,950,305
    kpss_data.rename(columns={'filing_date':'filing_date_kpss','issue_date':'issue_date_kpss'},inplace=True)

    metadata=metadata[['patent_id', 'grant_date', 'grant_year',
                       'num_claims', 'section_id','subsection_id',
                       'group_id', 'subgroup_id', 'category',
                       'application_id', 'filing_date', 'filing_year',
                       'cpc_4']]

    metadata_1=pd.merge(metadata,concatenated_df,on=['patent_id']) # 6,708,963
    citations_data_1=pd.merge(metadata_1,forward_citations,on=['patent_id'])  # 4,868,707
    citations_data_2=pd.merge(citations_data_1,backward_citations,on=['patent_id'],how='left')  # 4,868,707 (actual merge 4694128)
    kpss_data_1=pd.merge(citations_data_2,kpss_data,left_on=['patent_id'],right_on=['patent_num'],how='left')  # 4,868,707
    # 1,794,915 (actual merge)
    kpss_data_1.to_csv(data_dir+'compiled_measures_lagged'+str(lag)+'.csv',index=False)

    return kpss_data_1
