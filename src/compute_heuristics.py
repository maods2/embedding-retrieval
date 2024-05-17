import numpy as np
import torch
import pandas as pd
import pickle

def load_embeddings(pickle_file_path):
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_data_dict = pickle.load(pickle_file)

    # Convert lists to numpy arrays
    data = loaded_data_dict["embedding"]
    labels = np.array(loaded_data_dict["target"])
    return data, labels, loaded_data_dict

def apply_heuristics(embedding):
    # index order
    # hiper      -> 0
    # membran    -> 1
    # sclero     -> 2
    # normal     -> 3
    # podoc      -> 4
    # cresc      -> 5 
    # sclero seg -> 6
    emb = embedding.copy()
    for i in range(len(emb)):
        
        # Heirstics I
        # In case we have Normal class as positive, all other classes should be 0
        if emb[i][3] > 0.5:
            emb[i][0] = 0
            emb[i][1] = 0
            emb[i][2] = 0
            emb[i][4] = 0
            emb[i][5] = 0
            emb[i][6] = 0

        # Heirstics II
        # In case sclerose classifier is negative the values from sclerosis segmenter should be 0
        if emb[i][2] <= 0.5:
            emb[i][6] = 0

    return emb

if __name__ == "__main__":


    path_train = './embeddings/semantic_train.pickle'
    path_test = './embeddings/semantic_test.pickle'
    _, _, semant_result_train = load_embeddings(path_train)
    _, _, semant_result_test = load_embeddings(path_test)


    semant_result_train['embedding'] = apply_heuristics(semant_result_train['embedding'])
    semant_result_test['embedding'] = apply_heuristics(semant_result_test['embedding'])
    # print(semant_result_test['embedding'])

    with open(path_train, 'wb') as pickle_file:
        pickle.dump(semant_result_train, pickle_file)

    with open(path_test, 'wb') as pickle_file:
        pickle.dump(semant_result_test, pickle_file)

