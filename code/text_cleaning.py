import pandas as pd
from utility import clean_files
from . import configs

'''
1. Uses Patent Title and Abstract as input (OnlyPatentText-CPC.csv) for cleaning - 
Iteration 1 - Removes special characters, extra spaces and lines, stop words and lemmatises each word 
Iteration 2 - Eliminates words with frequency count < 3 (across all patent documents) and corpus-specific s
top words (very frequent words) like system, method, etc. 

Output files generated - 
1. FreqDist-CPC.csv
2. CleanedPatentText-CPC-Iter1.csv
3. CleanedTextData-CPC-Iter2.csv
'''

project_dir= configs.project_dir
data_dir= configs.data_dir
stats_dir= configs.stats_dir


def clean_patent_text():
    textdata=pd.read_csv(data_dir+'OnlyPatentText-CPC.csv')  # 6,779,677
    textdata['patent_id']=textdata['patent_id'].astype('str')
    ids=textdata['patent_id'].tolist()
    textdata['text']=textdata['title'].astype('str')+' '+textdata['abstract'].astype('str')
    corpus=textdata['text'].tolist()
    textdata['text']=textdata['text'].astype('str')

    print('File loaded ', len(corpus))

    cleanedcorpus_1 = clean_files.lemmatise_text(corpus)  # 6,779,673
    word_freq = get_word_distribution(cleanedcorpus_1)

    threshold=3
    corpus_stop_words=['make','include','process','method','use','apparatus','tool','system','device','involve',
                       'herein','wherein','whereas','body','produce','therefor','sub''group','layer','use','comprise',
                        'receive','system','provide','part','determine','item','line','say','mean','select','with','are',
                       'arrive','prefer','devise','vegf','lot','these','tee','later','jobs','effectively','finer',
                       'systems', 'methods','comprise','described','mechanism','plan','removal','remove','variety',
                       'development','step','control','variety','mechanism','send']

    cleanedcorpus = clean_files.eliminate_stop_words(cleanedcorpus_1, word_freq, corpus_stop_words, threshold)

    df2 = pd.DataFrame()
    df2['patent_id'] = ids
    df2['cleaned_text'] = cleanedcorpus
    df2['cleaned_text'] = df2['cleaned_text'].astype('str')
    df2=df2[df2['cleaned_text']!='']  # 6779661
    df2=df2[df2['cleaned_text']!='nan']  # 6779661
    df2['text_length']=df2['cleaned_text'].apply(lambda x: len(str(x).split(' ')) if str(x) != 'nan' else 0)
    df3=df2[df2.text_length!=0]  # 6779661
    df3['patent_id']=df3['patent_id'].astype('str')
    df3=df3.drop_duplicates(subset=['patent_id'],keep='first')  # 6779661
    df3=df3.reset_index(drop=True)

    df3.to_csv(data_dir+'CleanedTextData-CPC-Iter2.csv',index=False)


def get_word_distribution(corpus):
    word_freq = clean_files.frequency_distribution(corpus)  # 413,110
    freq_dist=pd.DataFrame()
    freq_dist['word']=list(word_freq.keys())
    freq_dist['count']=list(word_freq.values())
    freq_dist.to_csv(stats_dir+'FreqDist.csv',index=False)

    return word_freq


'''
Saving Iteration-1 stats - 
df1=pd.DataFrame()
df1['ids']=ids
df1['cleaned_text']=cleanedcorpus_1
df1['cleaned_text'] = df1['cleaned_text'].astype('str')
df1=df1[df1['cleaned_text']!='']
df1=df1[df1['cleaned_text']!='nan']
df1['text_length']=df1['cleaned_text'].apply(lambda x: len(str(x).split(' ')) if str(x) != 'nan' else 0)
df_1=df1[df1.text_length!=0]  # 6,779,673
df_1.to_csv(data_dir+'CleanedPatentText-CPC-Iter1.csv',index=False)
'''