# To use the CNNDM version BARTScore
from bart_score import BARTScorer

# score = bart_scorer.score(['This is interesting.'], ['This is fun.'],
#                           batch_size=16)  # generation scores from the first list of texts to the second list of texts.
# print(score, bart_scorer.score(['This is fun.'], ['This is interesting.'], batch_size=16))

import json
from tqdm import tqdm

f = open('14model_outputs.jsonl', 'r')
new_f = open('bartscore_14model_outputs.jsonl', 'w')
total_list = []

bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')  # cuda:0

for i, line in enumerate(tqdm(f)):
    inst = json.loads(line)

    ref = inst['summ']
    revs_dict = inst['revs']
    refs = [ref] * 14

    new_inst = {'revs': revs_dict, 'summ': ref, 'case': inst['case']}

    article = []
    for j in range(8):
        article.append(revs_dict['rev' + str(j + 1)])
    article = ' '.join(article)

    articles = [article] * 14

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

    hypo2ref_scores = bart_scorer.score(srcs=hypos, tgts=refs, batch_size=16)
    ref2hypo_scores = bart_scorer.score(srcs=refs, tgts=hypos, batch_size=16)
    arti2hypo_scores = bart_scorer.score(srcs=articles, tgts=hypos, batch_size=16)

    for j in range(14):  # 14 models
        key = sorted_keys[j]
        new_model_outputs[key] = {'model_summ': hypos[j], 'bart_score': {'hypo2ref': hypo2ref_scores[j],
                                                                         'ref2hypo': ref2hypo_scores[j],
                                                                         'arti2hypo': arti2hypo_scores[j]}}

    new_inst['model_output'] = new_model_outputs
    new_f.write(json.dumps(new_inst) + '\n')

new_f.close()
