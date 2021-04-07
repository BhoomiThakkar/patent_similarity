import pandas as pd
import numpy as np


# Calculate cosine similarities
def cos_sim(a, b):
    dot_product = np.dot(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return dot_product / (norm_a * norm_b)


def similarity(X, dest, map, ids, ipc):
    scores = []
    ids1 = []
    ids2 = []
    n_ids = len(ids)
    for i in range(n_ids):
        id1 = ids[i]
        index1 = map.loc[map['id']==id1].index[0]

        arr1 = X[index1]

        for j in range(n_ids):
            id2 = ids[j]
            index2 = map.loc[map['id'] == id2].index[0]
            arr2 = X[index2]

            score = cos_sim(arr1, arr2)
            scores.append(score)
            ids1.append(ids[i])
            ids2.append(ids[j])

        if i % 1000 == 0 and i != 0:
            print('\tCheckpoint: {}/{}'.format(i, n_ids))

    # print('\tPreparing dataframe and writing to file')
    df = pd.DataFrame()
    df['id1'] = ids1
    df['id2'] = ids2
    df['score'] = scores
    df = df.sort_values(by=['id1', 'score'], ascending=[True, False])
    df.to_csv(dest + str(ipc) + '.csv', index=False)





