import json
from tqdm import tqdm
import math
import os

if not os.path.exists('../autometric_scores/geval'):
    os.mkdir('../autometric_scores/geval')
new_f = open('../autometric_scores/geval/text_ada_001_geval_score.jsonl', 'w')
f = open('text_ada_001_geval_score_mod.jsonl', 'r')

dimensions = ['Aspect Relevance', 'Self-Coherence', 'Sentiment Consistency', 'Readability']
for line in tqdm(f):
    inst = json.loads(line)
    gpt_reply = inst['gpt_reply']
    case = inst['case']
    model_names = list(gpt_reply.keys())
    model_names.sort()
    model_output_dict = {model_name: None for model_name in model_names}
    raw_score_dict = {model_name: None for model_name in model_names}
    for model_name in model_names:
        model_geval_score_dict = {dim: None for dim in dimensions}
        model_geval_score_dict.update({'normalized_' + dim: None for dim in dimensions})

        model_reply = gpt_reply[model_name]
        dim_dict = {dim: None for dim in dimensions}
        for dim in dimensions:
            dim_reply = model_reply[dim]
            log_score = {1: None, 2: None, 3: None, 4: None, 5: None}
            dim_score = 0
            total_prob = 0
            for score in [1, 2, 3, 4, 5]:
                score_reply = dim_reply[str(score)]["logprobs"]
                score_log_prob = score_reply["token_logprobs"][-1]
                assert score_reply["tokens"][-1] == ' ' + str(score)
                assert isinstance(score_log_prob, float)
                log_score[score] = score_log_prob
                prob = math.exp(score_log_prob)
                assert 0 <= prob <= 1
                dim_score += score * prob
                total_prob += prob

            normalized_dim_score = dim_score / total_prob

            dim_dict[dim] = log_score
            model_geval_score_dict[dim] = dim_score
            model_geval_score_dict['normalized_' + dim] = normalized_dim_score
        model_output_dict[model_name] = {'model_summ': None, 'geval': model_geval_score_dict}
        raw_score_dict[model_name] = dim_dict
    new_f.write(json.dumps({'model_output': model_output_dict, 'case': case, 'raw_score': raw_score_dict})+'\n')
