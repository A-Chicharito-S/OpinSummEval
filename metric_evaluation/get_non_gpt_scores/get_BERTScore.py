import json
from tqdm import tqdm

from bert_score import BERTScorer
scorer = BERTScorer(lang="en", rescale_with_baseline=True)

f = open('14model_outputs.jsonl', 'r')
new_f = open('bertscore_14model_outputs.jsonl', 'w')
total_list = []

for i, line in enumerate(tqdm(f)):
    inst = json.loads(line)

    ref = inst['summ']
    refs = [ref] * 14

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
        hypos.append(model_outputs_dict[key])

    assert len(hypos) == 14

    P, R, F1 = scorer.score(hypos, refs)
    P = P.numpy().tolist()
    R = R.numpy().tolist()
    F1 = F1.numpy().tolist()

    for i in range(14):  # 14 models
        key = sorted_keys[i]
        new_model_outputs[key] = {'model_summ': hypos[i], 'bert_score': {'precision': P[i], 'recall': R[i], 'f1': F1[i]}}

    new_inst['model_output'] = new_model_outputs
    new_f.write(json.dumps(new_inst) + '\n')

new_f.close()
