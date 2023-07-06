from blanc import BlancHelp, BlancTune
import json
from tqdm import tqdm

f = open('14model_outputs.jsonl', 'r')
new_f = open('blanc_14model_outputs.jsonl', 'w')
total_list = []
blanc_help = BlancHelp(device='cuda', inference_batch_size=128)
blanc_tune = BlancTune(device='cuda', inference_batch_size=24, finetune_mask_evenly=False, finetune_batch_size=24)

for i, line in enumerate(tqdm(f)):
    inst = json.loads(line)

    ref = inst['summ']
    revs_dict = inst['revs']

    article = []
    for j in range(8):
        article.append(revs_dict['rev' + str(j + 1)])
    article = ' '.join(article)

    articles = [article] * 14

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

    scores_list_help = blanc_help.eval_pairs(docs=articles, summaries=hypos)
    scores_list_tune = blanc_tune.eval_pairs(docs=articles, summaries=hypos)

    assert len(scores_list_help) == 14
    assert len(scores_list_tune) == 14
    for j, key in enumerate(sorted_keys):
        new_model_outputs[key] = {'model_summ': hypos[j], 'blanc_score': {'help': scores_list_help[j], 'tune': scores_list_tune[j]}}

    new_inst['model_output'] = new_model_outputs
    new_f.write(json.dumps(new_inst) + '\n')

new_f.close()


# document = "Jack drove his minivan to the bazaar to purchase milk and honey for his large family."
# summary = "Jack bought milk and honey."
# blanc_help = BlancHelp(device='cuda', inference_batch_size=128)
# blanc_tune = BlancTune(device='cuda', inference_batch_size=24, finetune_mask_evenly=False, finetune_batch_size=24)
# documents = ["Jack drove his minivan to the bazaar to purchase milk and honey for his large family.",
#              "As Jill started taking a walk in the park, she certainly noticed that the trees were extra green this "
#              "year."]
# summaries = ["Jack bought milk and honey.",
#              "Jill saw green trees in the park."]
# tune_score = blanc_tune.eval_pairs(documents, summaries)
# help_score = blanc_help.eval_pairs(documents, summaries)
# # [0.2222222222222222, 0.0]
#
# print(tune_score)
# print(help_score)
