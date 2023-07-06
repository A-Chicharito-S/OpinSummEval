from tqdm import tqdm
import json
import rouge


def get_rouge_score(gold_sums, pred_sums, rouge_eval=None):
    if rouge_eval is None:
        rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                 max_n=2,
                                 limit_length=False,
                                 apply_avg=True,
                                 apply_best=False,
                                 alpha=0.5,  # Default F1_score
                                 stemming=True)

    scores = rouge_eval.get_scores(pred_sums, gold_sums)

    rouge_l = scores['rouge-l']['f']
    rouge_1 = scores['rouge-1']['f']
    rouge_2 = scores['rouge-2']['f']

    return rouge_1, rouge_2, rouge_l


# gold = ['I like the service. the rooms are very clean']
# pred1 = ['I like the service. rooms the very are clean']
# pred2 = ['The service is great and our room is clean']
# print(calculate(gold, pred1))
# print(calculate(gold, pred2))

f = open('14model_outputs.jsonl', 'r')
new_f = open('rouge_14model_outputs.jsonl', 'w')

rouge_evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                              max_n=2,
                              limit_length=False,
                              apply_avg=True,
                              apply_best=False,
                              alpha=0.5,  # Default F1_score
                              stemming=True)
for i, line in enumerate(tqdm(f)):
    inst = json.loads(line)

    ref = inst['summ']
    new_inst = {'revs': inst['revs'], 'summ': ref, 'case': inst['case']}

    model_outputs_dict = inst['model_output']
    new_model_outputs = {}
    assert i + 1 == int(inst['case'])
    keys = list(model_outputs_dict.keys())
    keys.sort()

    assert len(keys) == 14

    sorted_keys = keys
    hypos = []
    for key in sorted_keys:
        hypo = model_outputs_dict[key]
        r1_f, r2_f, rL_f = get_rouge_score(gold_sums=[ref], pred_sums=[hypo], rouge_eval=rouge_evaluator)
        new_model_outputs[key] = {'model_summ': hypo, 'ppl_score': {'R1_F1': r1_f, 'R2_F1': r2_f, 'RL_F1': rL_f}}

    new_inst['model_output'] = new_model_outputs
    new_f.write(json.dumps(new_inst) + '\n')

new_f.close()
