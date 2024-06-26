import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import pickle
import pandas as pd
from pathlib import Path
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt


def load_embeddings(pickle_file_path):
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_data_dict = pickle.load(pickle_file)

    # Convert lists to numpy arrays
    data = loaded_data_dict["embedding"]
    labels = np.array(loaded_data_dict["target"])
    return data, labels, loaded_data_dict

def run_queries(query_embedding, data_embeddings, similarity_function):
    similarity_scores = similarity_function(query_embedding.reshape(1, -1), data_embeddings).flatten()
    sorted_indices = np.argsort(similarity_scores)
    return sorted_indices, similarity_scores


def hamming(query_embedding, data_embeddings):
    distances = pairwise_distances(query_embedding, data_embeddings, metric='hamming').flatten()
    return distances

def jaccard(query_embedding, data_embeddings):
    distances = pairwise_distances(query_embedding, data_embeddings, metric='jaccard').flatten()
    return distances

def calculate_ap_at_k(relevant_labels, k_value):
    ap_num = 0
    tp = np.sum(relevant_labels[:k_value])
    for k in range(1, k_value + 1):
        tp_at_k = np.sum(relevant_labels[:k])
        precision_at_k =  tp_at_k / k
        # calculate numerator value for ap
        ap_num += precision_at_k * relevant_labels[k - 1]
        # print(f"P@{k+1}_{i+1} = {round(precision_at_k,2)}")

    ap_q = ap_num / tp if tp > 0 else 0
    return ap_q

def calculate_ar_at_k(relevant_labels, k_value, positives):
    ar_num = 0
    tp = np.sum(relevant_labels[:k_value])
    for k in range(1, k_value + 1):
        tp_at_k = np.sum(relevant_labels[:k])
        recall_at_k = tp_at_k / positives if positives > 0 else 0
        
        ar_num += recall_at_k * relevant_labels[k - 1]

    ar_q = ar_num / tp if tp > 0 else 0
    return ar_q

def calculate_f1_at_k(relevant_labels, k_value, positives):
    f1_num = 0
    tp = np.sum(relevant_labels[:k_value])
    for k in range(1, k_value + 1):
        tp_at_k = np.sum(relevant_labels[:k])
        precision_at_k =  tp_at_k / k
        recall_at_k = tp_at_k / positives if positives > 0 else 0
        f1_score_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) \
            if (precision_at_k + recall_at_k) > 0 else 0
        
        f1_num += f1_score_at_k * relevant_labels[k - 1]

    af1_q = f1_num / tp if tp > 0 else 0
    return af1_q

def evaluate_retrieval_pipeline(
        data_embeddings,
        data_labels, 
        query_embeddings, 
        query_labels, 
        k,
        similarity_function
        ):
    
    ap_at_k_list = []
    ar_at_k_list = []
    af1_at_k_list = []
    for q in range(len(query_embeddings)):
        query_label = query_labels[q] 
        sorted_indices, _ = run_queries(query_embeddings[q], data_embeddings, similarity_function)
        sorted_labels = data_labels[sorted_indices]
        # [1,0,1,1,1,0] relevant label = 1
        relevant_labels = sorted_labels == np.full((len(sorted_labels),), query_label)
        total_positives = np.sum(data_labels == query_label)
        average_precision = calculate_ap_at_k(relevant_labels, k)
        # average_recall = calculate_ar_at_k(relevant_labels, k, positives=total_positives)
        # average_f1score = calculate_f1_at_k(relevant_labels, k, positives=total_positives)
        ap_at_k_list.append(average_precision)
        # ar_at_k_list.append(average_recall)
        # af1_at_k_list.append(average_f1score)


    map_at_k = sum(ap_at_k_list) / len(query_embeddings)
    # mar_at_k = sum(ar_at_k_list) / len(query_embeddings)
    # maf1_at_k = sum(af1_at_k_list) / len(query_embeddings)
    
    print(f"mAP@{k} = {round(map_at_k, PRECISION)}")
    # print(f"mAR@{k} = {round(mar_at_k, PRECISION)}")
    # print(f"mAF1@{k} = {round(maf1_at_k, PRECISION)}")

    return round(map_at_k, PRECISION), 0, 0

def dot_product(query_embedding: np.ndarray, data_embeddings: np.ndarray) -> np.ndarray:
    distances = np.dot(data_embeddings, np.transpose(query_embedding)).flatten()
    return distances

def euclidian(query_embedding, data_embeddings):
    distances = euclidean_distances(query_embedding, data_embeddings).flatten()
    return distances

def cosine_similarity_func(query_embedding, data_embeddings):
    similarity = cosine_similarity(query_embedding, data_embeddings).flatten()
    distances = 1 - similarity
    return distances


def normalize_with_L2_norm(vector):
    """
    Function to normalize a vector using the L2 (Euclidean) norm.

    Parameters:
        vector (numpy.ndarray): The vector to be normalized.

    Returns:
        numpy.ndarray: The normalized vector.
    """
    norm = np.linalg.norm(vector)
    normalized_vector = vector / norm
    return normalized_vector

def transform_matrix(matrix):
    transformed_matrix = np.where(matrix > 0.51, 1, 0)
    return transformed_matrix

def bar_plot(ax: plt.Axes, 
             data: pd.DataFrame, 
             metric: str, 
             group_names: list[str],
             xlabel: str | None = None,
             xticks: list | None = None,
             colors=None, 
             total_width=0.8, 
             single_width=1, 
             legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: pandas.DataFrame

    metric: metric to be considered
        will plot the data found in column metric.

    group_names: names for each element of the group
        group_names[i] is the name on the legend for the i-th element of each group 

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """
    # we want to build a dict-like object
    # such that d[model] = [metric@k1 for model, metric@k2 for model, ...]

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(group_names)
    
    # The width of a single bar
    bar_width = total_width / n_bars
    bar_handles = []

    # Iterate over all data
    for i, group_name in enumerate(group_names):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        group_data = data[data['model'] == group_name]

        # Draw a bar for every value of that type
        for x, (value, k) in enumerate(zip(group_data[metric], group_data['k'])):
            bar = ax.bar(x + x_offset, value, width=bar_width * single_width, color=colors[i % len(colors)])
            bar.set_label(k)
            if x + 1 == len(group_data[metric]):
                bar_handles.append(bar[0])
            ax.text(x + x_offset, value , str(round(value,2)), ha='center', fontsize=7)
    
    if legend:
        # ax.legend(bar_handles, group_names)
        ax.legend(bar_handles, group_names, bbox_to_anchor=(1.05, 1), loc='upper left')

    if xticks is not None:
        ax.set_xticks(range(len(xticks)), xticks)
        
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    # for bar in ax.patches:
    #     bar.set_width(bar.get_width() * 1.5)  # Aumenta a largura em 1.5 vezes

def plot_metric(metric: str, title: str):
    fig, axs = plt.subplots(len(DISTANCES), figsize=(15, 15))
    for i, distance in enumerate(DISTANCES):
        data = df[df['distance'] == distance]
        bar_plot(axs[i], data, metric, 
                group_names=MODELS,
                xticks=K_LIST,
                xlabel='k', 
                total_width=0.8)

        axs[i].set_title(distance)
        axs[i].set_xlabel('k')
        axs[i].set_ylabel(metric)
        axs[i].set_ylim(0.0, 1.0)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'./embeddings/{title}.png')
    plt.savefig(f'./embeddings/{title}.svg')
    plt.show()

if __name__ == "__main__":

 
    K_LIST = [1 , 5, 10, 15, 30, 50] # k is the value of retrieval images per query to be evaluated on the MAP@k
    PRECISION = 2  # number of decimal digits of precision

    
    path_test = './embeddings/efficientnetb0_4096_autoencoder_test.pickle'
    path_train = './embeddings/efficientnetb0_4096_autoencoder_train.pickle'
    encod_data_test, encod_labels_test, _ = load_embeddings(path_test)
    encod_data_train, encod_labels_train, _ = load_embeddings(path_train)

    path_test = './embeddings/efficientnetb0_4096_pretrained_test.pickle'
    path_train = './embeddings/efficientnetb0_4096_pretrained_train.pickle'
    triplet_data_test, triplet_labels_test, _ = load_embeddings(path_test)
    triplet_data_train, triplet_labels_train, _ = load_embeddings(path_train)

    path_test = './embeddings/efficientnet_SwaV_test.pickle'
    path_train = './embeddings/efficientnet_SwaV_train.pickle'
    swav_data_test, swav_labels_test, _ = load_embeddings(path_test)
    swav_data_train, swav_labels_train, _ = load_embeddings(path_train)

    path_test = './embeddings/semantic_test.pickle'
    path_train = './embeddings/semantic_train.pickle'
    semant_data_test, semant_labels_test, _ = load_embeddings(path_test)
    semant_data_train, semant_labels_train, _ = load_embeddings(path_train)

    #############################################################

    path_test = './embeddings/semantic_att_efficientnetb0_encoder_test.pickle'
    path_train = './embeddings/semantic_att_efficientnetb0_encoder_train.pickle'
    semant_encod_data_test, semant_encod_labels_test, _ = load_embeddings(path_test)
    semant_encod_data_train, semant_encod_labels_train, _ = load_embeddings(path_train)

    path_test = './embeddings/semantic_att_efficientnetb0_test.pickle'
    path_train = './embeddings/semantic_att_efficientnetb0_train.pickle'
    semant_triplet_data_test, semant_triplet_labels_test, _ = load_embeddings(path_test)
    semant_triplet_data_train, semant_triplet_labels_train, _ = load_embeddings(path_train)

    path_test = './embeddings/semantic_att_efficientnet_SwaV_test.pickle'
    path_train = './embeddings/semantic_att_efficientnet_SwaV_train.pickle'
    semant_swav_data_test, semant_swav_labels_test, _ = load_embeddings(path_test)
    semant_swav_data_train, semant_swav_labels_train, _ = load_embeddings(path_train)


    # Transforming semantic attributes to binary to be tested as well
    semant_binary_data_train = transform_matrix(semant_data_train)
    semant_binary_data_test = transform_matrix(semant_data_test)

    # Normalize embeddings 
    encod_data_train = normalize_with_L2_norm(encod_data_train)
    encod_data_test = normalize_with_L2_norm(encod_data_test)

    triplet_data_train = normalize_with_L2_norm(triplet_data_train)
    triplet_data_test = normalize_with_L2_norm(triplet_data_test)

    swav_data_train = normalize_with_L2_norm(swav_data_train)
    swav_data_test = normalize_with_L2_norm(swav_data_test)

    semant_data_train = normalize_with_L2_norm(semant_data_train)
    semant_data_test = normalize_with_L2_norm(semant_data_test)


    semant_encod_data_train = normalize_with_L2_norm(semant_encod_data_train)
    semant_encod_data_test = normalize_with_L2_norm(semant_encod_data_test)

    semant_triplet_data_train = normalize_with_L2_norm(semant_triplet_data_train)
    semant_triplet_data_test = normalize_with_L2_norm(semant_triplet_data_test)

    semant_swav_data_train = normalize_with_L2_norm(semant_swav_data_train)
    semant_swav_data_test = normalize_with_L2_norm(semant_swav_data_test)

    emb_bundle = [
        ('autoenconder', encod_data_train, encod_labels_train, encod_data_test, encod_labels_test),
        ('triplet', triplet_data_train, triplet_labels_train, triplet_data_test, triplet_labels_test),
        ('swav', swav_data_train, swav_labels_train, swav_data_test, swav_labels_test),
        ('semant_att', semant_data_train, semant_labels_train, semant_data_test, semant_labels_test),
        ('semant_att_bin', semant_binary_data_train, semant_labels_train, semant_binary_data_test, semant_labels_test),
        
        ('semant_autoenconder', semant_encod_data_train, semant_encod_labels_train, semant_encod_data_test, semant_encod_labels_test),
        ('semant_triplet', semant_triplet_data_train, semant_triplet_labels_train, semant_triplet_data_test, semant_triplet_labels_test),
        ('semant_swav', semant_swav_data_train, semant_swav_labels_train, semant_swav_data_test, semant_swav_labels_test),

    ]

    res = {
            "model":[],
            "distance":[],
            "mAP@":[],
            # "mAR@":[],
            # "mAF1@":[],
            "k":[],
            }

    for m_name, x_train, y_train, x_test, y_test in emb_bundle:
            print(f"For embedding '{m_name}'".center(80, '-'))
            for k in K_LIST:
                    print("\nCosine Similarity ")
                    map, mar, maf1 = evaluate_retrieval_pipeline(
                            data_embeddings=x_train,
                            data_labels=y_train, 
                            query_embeddings=x_test, 
                            query_labels=y_test, 
                            k=k,
                            similarity_function=cosine_similarity_func
                            )
                    res["model"].append(m_name)
                    res["distance"].append("cosine similarity")
                    res["mAP@"].append(map)
                    res["k"].append(k)

                    print("\nEuclidian Distance ")
                    map, mar, maf1 = evaluate_retrieval_pipeline(
                            data_embeddings=x_train,
                            data_labels=y_train, 
                            query_embeddings=x_test, 
                            query_labels=y_test, 
                            k=k,
                            similarity_function=euclidian
                    )
                    res["model"].append(m_name)
                    res["distance"].append("euclidian")
                    res["mAP@"].append(map)
                    res["k"].append(k)
                    print('-'*80)

                    print("\Jaccard Distance ")
                    map, mar, maf1 = evaluate_retrieval_pipeline(
                            data_embeddings=x_train,
                            data_labels=y_train, 
                            query_embeddings=x_test, 
                            query_labels=y_test, 
                            k=k,
                            similarity_function=jaccard
                    )
                    res["model"].append(m_name)
                    res["distance"].append("jaccard")
                    res["mAP@"].append(map)
                    res["k"].append(k)
                    print('-'*80)

                    print("\hamming Distance ")
                    map, mar, maf1 = evaluate_retrieval_pipeline(
                            data_embeddings=x_train,
                            data_labels=y_train, 
                            query_embeddings=x_test, 
                            query_labels=y_test, 
                            k=k,
                            similarity_function=hamming
                    )
                    res["model"].append(m_name)
                    res["distance"].append("hamming")
                    res["mAP@"].append(map)
                    res["k"].append(k)
                    print('-'*80)

    # Saving the results in csv
    df = pd.DataFrame(res)
    df = df.sort_values(by=['model','distance','k','mAP@'], ascending=[False,False,True,False]).reset_index(drop=True)
    df.to_csv('./embeddings/query_same_dataset.csv')

    # Plotting results

    # embeddings to be evaluated
    MODELS = ['autoenconder', 'triplet', 'swav', 'semant_att','semant_att_bin', 'semant_autoenconder', 'semant_triplet', 'semant_swav']
    METRICS = ['mAP@']
    DISTANCES = ['euclidian', 'cosine similarity', 'jaccard','hamming']
    plot_metric('mAP@', "Mean Avarage Precision @k")