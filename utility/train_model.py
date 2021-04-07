# Import all the dependencies
import csv
import multiprocessing
from utility import calculate_cosine
from nltk.tokenize import word_tokenize
from code import configs
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

'''
# Algorithm:
1. Parameter setting for model
2. defining model specifications in train_model
3. checking similar tags and model accuracy (optional0
'''

cores=multiprocessing.cpu_count()
max_epochs= configs.max_epochs
vec_size= configs.vec_size
SEED= configs.SEED

print('Epochs = {}, Cores = {}, Vector-size = {}'.format(max_epochs, cores, vec_size))


def train_model(data, ids, destination_dir, alpha, type1=''):

    print('\tTagging data .. ')
    # returns a list of (tokens, id)
    tagged_data = [TaggedDocument(words=word_tokenize(str(_d).lower()), tags=[str(ids[i])]) for i, _d in enumerate(data)]

    print('\tPreparing model with the following parameters: epochs = {}, vector_size = {}, alpha = {} .. '.
          format(max_epochs, vec_size, alpha))

    model = Doc2Vec(vector_size=vec_size,
                    workers=cores//2,
                    alpha=alpha,  # initial learning rate
                    min_count=2,  # Ignore words having a total frequency below this
                    dm_mean=1,  # take mean of word2vec and utility
                    seed=SEED,
                    dm=1)  # PV-DM over PV-DBOW

    model.build_vocab(tagged_data, keep_raw_vocab=False, progress_per=10)

    print('\tBeginning model training .. ')

    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model.save(destination_dir + type1 + 'd2v.model')

    # print("Model Saved .. ")
    with open(destination_dir + type1 + 'model-log.txt', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['alpha', model.min_alpha])
        writer.writerow(['corpus documents', model.corpus_count])
        writer.writerow(['corpus words', model.corpus_total_words])
        writer.writerow(['epochs', model.epochs])
        writer.writerow(['window_size', model.window])

    print("\tModel Saved .. ")

    return model


def check_similarity(ids,model,tagged_data):

    for id1 in ids[:10]:
        arr1 = model.docvecs[str(id1)]
        ind1 = ids.index(int(id1))
        print(id1, tagged_data[ind1][0][:10])
        arr2 = model.docvecs.most_similar([arr1], topn=5)  # top 5 most similar documents
        for a2 in arr2:
            id2 = a2[0]
            ind2 = ids.index(int(id2))
            arr_1=model.docvecs[str(id2)]
            print('\t', id2, tagged_data[ind2][0][:10], calculate_cosine.cosine(arr1,arr_1))
