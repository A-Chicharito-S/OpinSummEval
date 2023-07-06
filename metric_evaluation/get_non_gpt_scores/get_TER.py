from sacrebleu.metrics import TER
import json
from tqdm import tqdm

f = open('14model_outputs.jsonl', 'r')
new_f = open('ter_score_14model_outputs.jsonl', 'w')
total_list = []
ter_score = TER()

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

    for key in sorted_keys:
        hypo = model_outputs_dict[key]
        score = ter_score.sentence_score(hypothesis=hypo, references=[ref]).score
        new_model_outputs[key] = {'model_summ': hypo, 'ter_score': score}

    new_inst['model_output'] = new_model_outputs
    new_f.write(json.dumps(new_inst) + '\n')

new_f.close()

# refs = [
#     ['It was not unexpected.', 'The man bit him first.'],
#     ['No one was surprised.', 'The man had bitten the dog.'],
# ]
# sys = ["It wasn't surprising.", 'The man had just bitten him.']
#
#
#
# scores = ter_score.corpus_score(hypotheses=sys, references=refs)
# # BLEU = 29.44 82.4/42.9/27.3/12.5 (BP = 0.889 ratio = 0.895 hyp_len = 17 ref_len = 19)
# print(scores)
#
# print(ter_score.sentence_score(hypothesis=sys[0], references=[refs[0][0], refs[1][0]]))
# print(ter_score.sentence_score(hypothesis=sys[1], references=[refs[0][1], refs[1][1]]))
#
# print(ter_score.corpus_score(hypotheses=[sys[0]], references=[[refs[0][0]], [refs[1][0]]]))
# print(ter_score.corpus_score(hypotheses=[sys[1]], references=[[refs[0][1]], [refs[1][1]]]))
