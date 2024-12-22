import numpy as np


def ndcg_k(reco_relevance, relevance, mask, k=10):
    reco_relevance = reco_relevance[mask]
    relevance = relevance[mask]

    v = np.asarray(relevance.sum(axis=1).flatten(), dtype=int)[0].clip(1, k)
    ideal_relevance = np.vstack([np.concatenate((np.ones(i), np.zeros(k - i))) for i in v])

    discount = 1 / np.log2(np.arange(2, k + 2))
    idcg = (ideal_relevance * discount).sum(axis=1)
    dcg = (reco_relevance * discount).sum(axis=1)
    ndcg = (dcg / idcg).mean()

    return ndcg
