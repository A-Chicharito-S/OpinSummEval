import openpyxl
import os
import numpy as np
from agreement_metrics import cohen_kappa, fleiss_kappa, krippendorff_alpha, Gwets_AC1
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_one_batch_score(anno_num, path):
    annot_batch = openpyxl.load_workbook('annotator' + anno_num + '/' + path)
    sheet_names = annot_batch.sheetnames
    one_batch_score_list = []  # should have 10 elements
    for sheet_name in sheet_names:
        sheet = annot_batch[sheet_name]
        one_sheet_score_list = []  # should have 14 elements
        for row in sheet['13:26']:
            one_model_score_list = []  # should have 4 elements
            for cell in row[:4]:
                one_model_score_list.append(cell.value)
            one_sheet_score_list.append(one_model_score_list)

        one_batch_score_list.append(one_sheet_score_list)
    batch_array = np.array(one_batch_score_list)  # shape: (10, 14, 4)
    return batch_array


path_list = os.listdir('final_batch/annot1')
cohen_kappa_total = 0
fleiss_kappa_total = 0
krippendorff_alpha_total = 0
gwets_AC1_total = 0

for i, path in tqdm(enumerate(path_list)):
    annot1_batch_arr = get_one_batch_score(anno_num='1', path=path)  # shape: (10, 14, 4)
    annot2_batch_arr = get_one_batch_score(anno_num='2', path=path)  # shape: (10, 14, 4)

    cohen_kappa_score = cohen_kappa(annot1=annot1_batch_arr, annot2=annot2_batch_arr)  # shape: (14, 4)
    fleiss_kappa_score = fleiss_kappa(annot1=annot1_batch_arr, annot2=annot2_batch_arr)  # shape: (14, 4)
    krippendorff_alpha_score = krippendorff_alpha(annot1=annot1_batch_arr, annot2=annot2_batch_arr)  # shape: (14, 4)
    gwets_AC1_score = Gwets_AC1(annot1=annot1_batch_arr, annot2=annot2_batch_arr)  # shape: (14, 4)

    cohen_kappa_score[np.isnan(cohen_kappa_score)] = 1
    # means this entry only has one category score, e.g., 4,4,4,4,4
    fleiss_kappa_score[np.isnan(fleiss_kappa_score)] = 1

    cohen_kappa_total += cohen_kappa_score.mean(axis=0)
    fleiss_kappa_total += fleiss_kappa_score.mean(axis=0)
    krippendorff_alpha_total += krippendorff_alpha_score.mean(axis=0)
    gwets_AC1_total += gwets_AC1_score.mean(axis=0)

print('cohen_kappa: {}'.format(cohen_kappa_total/len(path_list)))
print('fleiss_kappa: {}'.format(fleiss_kappa_total/len(path_list)))
print('gwets_AC1: {}'.format(gwets_AC1_total/len(path_list)))
print('krippendorff_alpha: {}'.format(krippendorff_alpha_total/len(path_list)))
