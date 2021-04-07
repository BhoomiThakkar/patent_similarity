from . import compute_scores, collate_regression_data, configs, aggregate_scores, text_cleaning, train_vectors

'''
1. Set variable and directories

Function calls -
2. to clean patent abstracts
3. Train Doc2Vec architecture to compute numeric vectors 
4. Calculate similarity between the patent vectors
5. Combine patent1, patent2 scores at the patent level
6. Run Regressions 
'''

lag= configs.lag
alpha= configs.alpha
ENDYEAR= configs.ENDYEAR
STARTYEAR= configs.STARTYEAR
project_dir= configs.project_dir

text_cleaning.clean_patent_text()
train_vectors.train_doc2vec(alpha, STARTYEAR, ENDYEAR)
compute_scores.compute_scores(lag, STARTYEAR, ENDYEAR)
concatenated_df= aggregate_scores.concatenate_year_wise(lag, STARTYEAR, ENDYEAR)
all_data= collate_regression_data.compile_regression_data(lag)
