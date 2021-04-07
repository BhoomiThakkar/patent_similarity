import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

'''
# Algorithm for Cosine Calculation in multiple ways
1. nXn Cosine Matrix using sklearn's module
2. Cosine Similarity for one set of IDs with other IDs in the same set
'''


# Cosine Matrix
def compute_cosine_matrix(currentmodel, ids1, previousmodel, ids2, filepath):
    arr1 = []
    arr2 = []

    # cosine similarity ..

    for id1 in ids1:
        a1 = currentmodel.docvecs[str(id1)]
        arr1.append(a1)

    for id2 in ids2:
        a2 = previousmodel.docvecs[str(id2)]
        arr2.append(a2)

    arr1 = np.asarray(arr1)
    arr1 = np.reshape(arr1, (len(ids1), 100))
    arr2 = np.asarray(arr2)
    arr2 = np.reshape(arr2, (len(ids2), 100))

    scores = cosine_similarity(arr1, arr2)

    scoredf = pd.DataFrame(data=scores,
                 index=ids1,
                 columns=ids2)

    scoredf.to_csv(filepath)


# Calculate cosine similarities
def cosine(a, b):
    dot_product = np.dot(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return dot_product / (norm_a * norm_b)


# Similarity between elements from the same set
def similarity(model, ids, filepath):
    scores = []
    ids1 = []
    ids2 = []
    n_ids = len(ids)

    for i in range(n_ids):
        key1 = ids[i]
        arr1 = model.docvecs[str(key1)]
        for j in range(n_ids):
            key2 = ids[j]
            arr2 = model.docvecs[str(key2)]
            score = cosine(arr1, arr2)
            scores.append(score)
            ids1.append(ids[i])
            ids2.append(ids[j])

    # print('\tPreparing dataframe and writing to file')
    df = pd.DataFrame()
    df['id1'] = ids1
    df['id2'] = ids2
    df['score'] = scores
    df = df.sort_values(by=['id1', 'score'], ascending=[True, False])
    df.to_csv(filepath, index=False)

