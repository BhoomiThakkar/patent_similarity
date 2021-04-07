import pandas as pd
from utility import clean_patent_number
from . import configs

'''
1. Collate  citations for patents using uspatentcitation.tsv from PatentsView
Forward Citation (Total and by Indiviudal category) and Backward Citations 
2. Output files generated are 
    1. applicant_forward_citations.csv
    2. examiner_forward_citations.csv
    3. no_forward_citations.csv
    4. remaining_forward_citations.csv
    5. total_backward_citations_dec20.csv
    6. total_forward_citations_dec20.csv
'''

# Collate citations data

project_dir= configs.project_dir
data_dir= configs.data_dir
destination_dir=data_dir

citations=pd.read_csv(data_dir+'uspatentcitation.tsv',sep='\t') # 113129077

# Backward Citations
back_citations=citations
back_citations['patent_id']=back_citations['patent_id'].apply(clean_patent_number.cleanpatent)  # 11,312,9077
back_citations_1=back_citations.dropna(subset=['patent_id']) # 11,312,9077
back_citations_1['patent_id']=back_citations_1['patent_id'].astype('str')
back_citations_2=back_citations_1[back_citations_1['patent_id']!='']  # 11,312,9077
back_citations_2['patent_id']=back_citations_2['patent_id'].astype('int64')
backward_citations=back_citations_2.groupby(by=['patent_id']).count()
backward_citations=backward_citations[['citation_id']]
backward_citations.rename(columns={'citation_id':'backward_citations'},inplace=True)
backward_citations.to_csv(destination_dir + 'total_backward_citations_dec20.csv')  # 7,074,837


# Forward citations
citations['citation_id']=citations['citation_id'].apply(clean_patent_number.cleanpatent)  # 113129077
citations_1=citations.dropna(subset=['citation_id'])  # 113129077
citations_1['citation_id']=citations_1['citation_id'].astype('str')
citations_2=citations_1[citations_1['citation_id']!='']  # 113129041
citations_2['citation_id']=citations_2['citation_id'].astype('int64')

total_citations=citations_2.groupby(by=['citation_id']).count() # 7,583,358
total_citations=total_citations[['patent_id']]
total_citations.rename(columns={'patent_id':'forward_citations'},inplace=True)
total_citations.rename(columns={'citation_id':'patent_id'},inplace=True)
total_citations.to_csv(destination_dir + 'total_forward_citations_dec20.csv')

applicants=citations_2[citations_2.category=='cited by applicant']
examiner=citations_2[citations_2.category=='cited by examiner']
other=citations_2[(citations_2.category!='cited by examiner')&(citations_2.category!='cited by applicant')]
print(len(applicants)+len(examiner)+len(other),len(citations_2))  # 113129041 113129041

df1=applicants # 41946968
df2=df1.groupby(by=['citation_id']).count()
df2=df2[['patent_id']]
df2=df2.reset_index()
df2.rename(columns={'patent_id':'application_added_forward_citations'},inplace=True)
df2.rename(columns={'citation_id':'patent_id'},inplace=True)
df2.to_csv(destination_dir+'applicant_forward_citations.csv')  # 3,787,745
print(len(df1),len(df2))

df1=examiner  # 23,883,675
df2=df1.groupby(by=['citation_id']).count()
df2=df2[['patent_id']]
df2=df2.reset_index()
df2.rename(columns={'patent_id':'examiner_added_forward_citations'},inplace=True)
df2.rename(columns={'citation_id':'patent_id'},inplace=True)
df2.to_csv(destination_dir+'examiner_forward_citations.csv')  # 5,468,842
print(len(df1),len(df2))

df1=other # 47298398
df2=df1.groupby(by=['citation_id']).count()
df2=df2[['patent_id']]
df2=df2.reset_index()
df2.rename(columns={'patent_id':'other_citations'},inplace=True)
df2.rename(columns={'citation_id':'patent_id'},inplace=True)
df2.to_csv(destination_dir+'remaining_forward_citations.csv') # 5071753
print(len(df1),len(df2))


