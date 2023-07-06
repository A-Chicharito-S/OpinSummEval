import json
from tqdm import tqdm
import re
import os


# matching idea:
# 1. give a dimension name
# 2. change all the words to lower letters
# 3. match its first occurrence
# 4. match sth. like: dim[some symbols here(probably":")] an integer
def get_dim_score(dim, reply):
    pattern = dim + '.*.[\s]*\d'
    # print('dim: {}'.format(dim))
    # print('reply: {}'.format(reply))
    match = re.findall(pattern=pattern, string=reply, flags=re.I)[0]
    score = int(re.findall(pattern=r'\d', string=match)[0])
    return score


f = open('../14model_outputs.jsonl', 'r')
handcraft_f = open('chatgpt_handcraft_score_14_onebyone_unstru.jsonl', 'r')
if not os.path.exists('../autometric_scores/chatgpt_handcraft_14_unstru'):
    os.mkdir('../autometric_scores/chatgpt_handcraft_14_unstru')
new_f = open('../autometric_scores/chatgpt_handcraft_14_unstru/chatgpt_score_14_unstru.jsonl', 'w')
dimensions = ['Relevance', 'Coherence', 'Consistency', 'Readability']
for line, f_line in tqdm(zip(handcraft_f, f)):
    inst = json.loads(line)
    gpt_reply = inst['gpt_reply']
    model_names = list(json.loads(f_line)['model_output'].keys())
    model_names.sort()
    score_dict = {model_name: None for model_name in model_names}
    for model_name in model_names:
        # model_name = model_names[i]
        model_reply = gpt_reply[model_name].lower()
        model_score_dict = {'model_summ': None, 'chatgpt_handcraft_14_unstru': {dim: 0 for dim in dimensions},
                            'model_scoring_flag': None}
        is_scored = False
        for dim in dimensions:
            if dim.lower() in model_reply and len(re.findall(pattern='\d', string=model_reply)) != 0:
                dim_score = get_dim_score(dim=dim, reply=model_reply)
                is_scored = True
            else:
                dim_score = 1  # treat it as the lowest score
            model_score_dict['chatgpt_handcraft_14_unstru'][dim] = dim_score
        if is_scored:
            model_score_dict['model_scoring_flag'] = 'all scored'
        else:
            model_score_dict['model_scoring_flag'] = model_reply
        score_dict[model_name] = model_score_dict
    new_f.write(json.dumps({'model_output': score_dict, 'case': inst['case']}) + '\n')
