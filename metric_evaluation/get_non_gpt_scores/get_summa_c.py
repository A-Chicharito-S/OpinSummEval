from summac.model_summac import SummaCConv
import json
from tqdm import tqdm

# sentence-level
model_conv_snt = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

# document-level
model_conv_doc = SummaCConv(models=["vitc"], bins='percentile', granularity="document", nli_labels="e", device="cuda", start_file="default", agg="mean")

f = open('14model_outputs.jsonl', 'r')
new_f = open('summac_14model_outputs.jsonl', 'w')
total_list = []

for i, line in enumerate(tqdm(f)):
    inst = json.loads(line)

    ref = inst['summ']
    revs_dict = inst['revs']

    article = []
    for j in range(8):
        article.append(revs_dict['rev' + str(j + 1)])
    article = ' '.join(article)

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

    for j, key in enumerate(sorted_keys):
        model_sum = hypos[j]
        score_snt = model_conv_snt.score([article], [model_sum])["scores"][0]
        score_doc = model_conv_doc.score([article], [model_sum])["scores"][0]
        new_model_outputs[key] = {'model_summ': model_sum, 'summac_score': {'snt': score_snt, 'doc': score_doc}}

    new_inst['model_output'] = new_model_outputs
    new_f.write(json.dumps(new_inst) + '\n')

new_f.close()
