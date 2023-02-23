import argparse
from tqdm import tqdm
import nmslib
import random
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(list_1, list_2):
    """
    Calculate the cosine similarity between two lists.
    """
    cos_sim = dot(list_1, list_2) / (norm(list_1) * norm(list_2))
    return cos_sim


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max

def create_vector_space(embedd_path, metadados_path):
    """
    Create a vector space from the embeddings.
    """
    df = pd.read_csv(metadados_path, sep='\t')
    df = df.reset_index(drop=True)

    embeddings = np.loadtxt(embedd_path)
    embeddings = np.nan_to_num(embeddings)

    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(embeddings)
    index.createIndex({'post': 2}, print_progress=True)

    return index, df

def read_eval_ds():
    """
    Read the evaluation dataset.
    """
    df = pd.read_csv("data/evaluation/eval_users.csv")
    df["gt_reclist"] = df["gt_reclist"].apply(eval)
    df["reclist"] = df["reclist"].apply(eval)
    
    return df

if __name__ == '__main__':
    """Convert a yelp dataset file from json to csv."""

    parser = argparse.ArgumentParser(
            description='Avaliação de Embeddings',
            )

    parser.add_argument('embeddings_path', type=str, help='Arquivo de embeddings')
    parser.add_argument('metadados_path', type=str, help='Arquivo de metadados')

    args = parser.parse_args()

    # Load Evaluation Dataset
    eval_users = read_eval_ds()

    # Create a vector space from the embeddings
    index, df = create_vector_space(args.embeddings_path, args.metadados_path)
    print(df.head())

    # Create a Map Index
    item_code_id = {i: code for i, code in enumerate(df['business_id'])}
    item_id_code = {code: i for i, code in enumerate(df['business_id'])}

    ndcg_5_list = []
    ndcg_10_list = []
    for i, row in tqdm(eval_users.iterrows()):
        context_list = [row.user_perfil]
        gt_list = row.gt_reclist
        reclist = row.reclist

        # create context array
        code_context  = [item_id_code[str(id)] for id in context_list] # mapping ids to codes for table lookup
        context_array = []

        for c in code_context:
            context_array.append(np.array(index[c]))
        context_array = np.array(context_array).mean(axis=0)
    
        # rank
        code_reclist  = [item_id_code[str(id)] for id in reclist] # mapping ids to codes for table lookup
        rank_bussiness_id = {}
        for bussiness_id in reclist:
            cos_sim = cosine_similarity(np.array(index[item_id_code[str(bussiness_id)]]), context_array)
            rank_bussiness_id[bussiness_id] = cos_sim
        
        # order list from dict and create a binary list
        rank_bussiness_id = {k: v for k, v in sorted(rank_bussiness_id.items(), key=lambda item: item[1], reverse=True)}
        reclist_ranked    = list(rank_bussiness_id.keys())

        binary_reclist = [1 if x in gt_list else 0 for x in reclist_ranked]

        ndcg_5_list.append(ndcg_at_k(binary_reclist, k=5))
        ndcg_10_list.append(ndcg_at_k(binary_reclist, k=10))

    print("\n\nAvaliação de Embeddings")
    print("Embeddings: ", args.embeddings_path)
    print("Total Users: ", len(eval_users))
    print("NDCG@5: ", np.mean(ndcg_5_list))
    print("NDCG@10: ", np.mean(ndcg_10_list))