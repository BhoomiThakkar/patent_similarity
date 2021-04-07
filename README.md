# Code Pipeline to Compute Similarity 

data_processing: Compiles PatentsView Patent Data such as Title, Abstract, CPC Class, Grant and Filing Year for 6.7 million patents 
collate_citations: Compile forward citations by category for granted patents
run: Combined Function call to clean patent text (title and abstract), training numeric vectors, computing cosine similarity between patent pairs, 
     aggregate scores at the patent level, combine with forward citations and patent value and run regression
text_cleaning: Pre-process patent text, remove stop words, lemmatise text, remove words based on frequency 
train_vectors: Train cleaned patent text using Doc2Vec
compute_scores: Compute cosine similarity between patent pairs. Scores are calculated such that we compute similarity of a focal patent to all patents 
                granted in the previous year and in the same CPC class as the focal patent
aggregate_scores: Aggregate scores at the patent level
collate_regression_data: Combine scores with citations and patent value 
trial_regressions: Run regression for Total Citation ~ Patent Similarity controlling for technology class and grant year

