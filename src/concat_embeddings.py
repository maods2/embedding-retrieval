
import numpy as np
import torch
import pandas as pd
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embeddings(pickle_file_path):
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_data_dict = pickle.load(pickle_file)

    # Convert lists to numpy arrays
    data = loaded_data_dict["embedding"]
    labels = np.array(loaded_data_dict["target"])
    return data, labels, loaded_data_dict

# train data
def concat_embeddings(
        embedding,
        semantic_att,
        mode
        ):
    path_s = semantic_att['paths']
    path_e = embedding['paths']

    X_s = semantic_att['embedding']
    X_e = embedding['embedding']

    num_att = X_s.shape[1] + X_e.shape[1]
    feature_embeddings = np.empty((0, num_att))


    for i, (sem, emb, x_s, x_e) in enumerate(zip(path_s, path_e, X_s, X_e)):
        if sem != emb:
            raise Exception(f'images are not the same: {sem} - {emb}')
        
        concatenated_array = np.concatenate((x_s, x_e))
        feature_embeddings = np.vstack((feature_embeddings, concatenated_array))
        print(f'{i+1} of {len(X_s)} embs | model: {embedding["model"]} ')


    model_name = semantic_att['model']+'_'+embedding['model']
    data_dict = {
        "model": model_name,
        "embedding":feature_embeddings,
        "target":semantic_att['target'],
        "paths": semantic_att['paths'],
        "classes":semantic_att['classes']
    }

    with open(f'./embeddings/{model_name}_{mode}.pickle', 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)


if __name__ == "__main__":

    emb_list = [
        (
            './embeddings/efficientnetb0_4096_autoencoder_train.pickle',
            './embeddings/efficientnetb0_4096_autoencoder_test.pickle'
        ),
        (
           './embeddings/efficientnetb0_4096_pretrained_train.pickle',
            './embeddings/efficientnetb0_4096_pretrained_test.pickle'
        ),
        (
           './embeddings/efficientnet_SwaV_train.pickle',
            './embeddings/efficientnet_SwaV_test.pickle'
        ),
    ]

    path_test = './embeddings/semantic_test.pickle'
    path_train = './embeddings/semantic_train.pickle'
    _, _, semant_result_test = load_embeddings(path_test)
    _, _, semant_result_train = load_embeddings(path_train)

    for path_train, path_test in emb_list:
    

        _, _, emb_result_train = load_embeddings(path_train)
        _, _, emb_result_test = load_embeddings(path_test)


        concat_embeddings(
                embedding=emb_result_train,
                semantic_att=semant_result_train,
                mode='train'
                )
        concat_embeddings(
                embedding=emb_result_test,
                semantic_att=semant_result_test,
                mode='test'
                )

