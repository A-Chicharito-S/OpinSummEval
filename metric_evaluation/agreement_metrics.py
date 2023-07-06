import sklearn.metrics as skmetric
import numpy as np
import statsmodels.stats.inter_rater as rater
import simpledorff
import pandas as pd
from pycm import pycm_obj


def cohen_kappa(annot1, annot2):
    # annotator1_round_2 & annotator2_round_2: (N, M, D)
    # N: # of instances;  M: # of models;  D: # of evaluation dimensions;
    N, M, D = annot1.shape
    one_batch_score = []
    for m in range(M):
        one_model_score = []
        for d in range(D):
            score = skmetric.cohen_kappa_score(y1=annot1[:, m, d], y2=annot2[:, m, d])
            one_model_score.append(score)
        one_batch_score.append(one_model_score)

    return np.array(one_batch_score)


def fleiss_kappa(annot1, annot2):
    # annotator1_round_2 & annotator2_round_2: (N, M, D)
    # N: # of instances;  M: # of models;  D: # of evaluation dimensions;
    N, M, D = annot1.shape
    one_batch_score = []
    for m in range(M):
        one_model_score = []
        for d in range(D):
            data, cate = rater.aggregate_raters(data=(np.array([annot1[:, m, d], annot2[:, m, d]]) - 1).T, n_cat=5)
            # .T is because the raters are assumed as columns
            # -1 is because the labels are: 0, 1, 2, ..., n_cat-1
            score = rater.fleiss_kappa(table=data, method='fleiss')
            one_model_score.append(score)
        one_batch_score.append(one_model_score)

    return np.array(one_batch_score)
    # returns a ndarray: (M, D)


def krippendorff_alpha(annot1, annot2):
    # annotator1_round_2 & annotator2_round_2: (N, M, D)
    # N: # of instances;  M: # of models;  D: # of evaluation dimensions;
    N, M, D = annot1.shape
    one_batch_score = []
    for m in range(M):
        one_model_score = []
        for d in range(D):
            data = []
            for n in range(N):
                data.append([n + 1, 'A', annot1[n, m, d]])
                # document_id, annotator_id, annotation
                data.append([n + 1, 'B', annot2[n, m, d]])
            df = pd.DataFrame(data, columns=['document_id', 'annotator_id', 'annotation'])
            score = simpledorff.calculate_krippendorffs_alpha_for_df(df, experiment_col='document_id',
                                                                     annotator_col='annotator_id',
                                                                     class_col='annotation')
            one_model_score.append(score)
        one_batch_score.append(one_model_score)

    return np.array(one_batch_score)
    # returns a ndarray: (M, D)


def Gwets_AC1(annot1, annot2):
    # annotator1_round_2 & annotator2_round_2: (N, M, D)
    # N: # of instances ;  M: # of models;  D: # of evaluation dimensions;
    N, M, D = annot1.shape
    one_batch_score = []
    for m in range(M):
        one_model_score = []
        for d in range(D):
            cm = pycm_obj.ConfusionMatrix(actual_vector=annot1[:, m, d], predict_vector=annot2[:, m, d])
            score = cm.AC1
            one_model_score.append(score)
        one_batch_score.append(one_model_score)

    return np.array(one_batch_score)

