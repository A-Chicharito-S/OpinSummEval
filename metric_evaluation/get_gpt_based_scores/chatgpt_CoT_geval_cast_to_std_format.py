import json
from tqdm import tqdm
import math
import os
import re


def get_dim_score(dim, reply):
    pattern1 = dim + '.*.[\s]*\d'
    pattern2 = 'Rating' + '.*.[\s]*\d'
    # print('dim: {}'.format(dim))
    # print('reply: {}'.format(reply))
    # print(reply)
    match = re.findall(pattern=pattern1, string=reply, flags=re.I)

    if len(match) == 0:
        match = re.findall(pattern=pattern2, string=reply, flags=re.I)
    match = match[0]
    score = int(re.findall(pattern=r'\d', string=match)[0])

    return score


dimensions = ['Aspect Relevance', 'Self-Coherence', 'Sentiment Consistency', 'Readability']


def sort_dim_score(dim):
    dim_path = 'geval_chatgpt_' + dim
    if not os.path.exists('../autometric_scores/' + dim_path):
        os.mkdir('../autometric_scores/' + dim_path)
    new_f = open('../autometric_scores/' + dim_path + '/chatgpt_geval_' + dim + '_score.jsonl', 'w')
    f = open('chatgpt_CoT_geval_score_mod.jsonl', 'r')
    for line in tqdm(f):
        inst = json.loads(line)
        gpt_reply = inst['gpt_reply']
        case = inst['case']
        model_names = list(gpt_reply.keys())
        model_names.sort()
        model_output_dict = {model_name: None for model_name in model_names}
        for model_name in model_names:
            model_geval_score_dict = {'model_summ': None, 'chatgpt_CoT_'+dim: 0, 'model_scoring_flag': None}

            model_reply = gpt_reply[model_name][dim]['content'].lower()
            is_scored = False

            if dim.lower() in model_reply and len(re.findall(pattern='\d', string=model_reply)) != 0:
                dim_score = get_dim_score(dim=dim, reply=model_reply)
                is_scored = True
            else:
                dim_score = 1  # treat it as the lowest score

            model_geval_score_dict['chatgpt_CoT_'+dim] = dim_score
            model_geval_score_dict['model_scoring_flag'] = 'scored' if is_scored else model_reply

            model_output_dict[model_name] = model_geval_score_dict
        new_f.write(json.dumps({'model_output': model_output_dict, 'case': case}) + '\n')

for dim in dimensions:
    print(dim)
    sort_dim_score(dim)
