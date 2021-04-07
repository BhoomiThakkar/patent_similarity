# Import all the dependencies
import pandas as pd
import multiprocessing
from utility import calculate_cosine
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

dir1='/Volumes/Elements/PatentScore-Dec27/'
cleaneddata=pd.read_csv(dir1+'CleanedTextData.csv')

year=2018
cleaneddata_1=cleaneddata[cleaneddata['grant_year']==year]
cleaneddata_1['patent_id']=cleaneddata_1['patent_id'].astype('str')
ids = cleaneddata_1['patent_id'].tolist()
data = cleaneddata_1['cleaned_text'].tolist()

tagged_data = [TaggedDocument(words=word_tokenize(str(_d).lower()), tags=[str(ids[i])]) for i, _d in enumerate(data)]

SEED=1013
alpha=0.05
vec_size=100
cores=multiprocessing.cpu_count()


model = Doc2Vec(vector_size=vec_size,
                workers=cores // 2,
                alpha=alpha,  # initial learning rate
                min_count=2,  # Ignore words having a total frequency below min_count
                dm_mean=1,  # take mean of word2vec and utility
                seed=SEED,
                dm=1)  # PV-DM over PV-DBOW

model.build_vocab(tagged_data, keep_raw_vocab=False, progress_per=10)
model.train(tagged_data,
            total_examples=model.corpus_count,
            epochs=model.epochs)

for id1 in ids[:10]:
    arr1 = model.docvecs[str(id1)]
    ind1 = ids.index(str(id1))
    print(id1, tagged_data[ind1][0][:10])

    arr2 = model.docvecs.most_similar([arr1], topn=5)  # top 5 most similar documents

    for a2 in arr2:
        id2 = a2[0]
        ind2 = ids.index(str(id2))
        arr_1 = model.docvecs[str(id2)]
        print(id2, tagged_data[ind2][0][:10])

        print('\t', id2, tagged_data[ind2][0][:20], calculate_cosine.cosine(arr1, arr_1))



