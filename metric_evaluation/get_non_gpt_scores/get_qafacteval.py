from qafacteval import QAFactEval
from tqdm import tqdm
import json
kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
          "verbose": True, "generation_batch_size": 32, \
          "answering_batch_size": 32, "lerc_batch_size": 8}

model_folder = "models"  # path to models downloaded with download_models.sh
metric = QAFactEval(
    lerc_quip_path=f"{model_folder}/quip-512-mocha",
    generation_model_path=f"{model_folder}/generation/model.tar.gz",
    answering_model_dir=f"{model_folder}/answering",
    lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
    lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
    **kwargs
)

# results = metric.score_batch_qafacteval(["This is a source document"], [["This is a summary."]], return_qa_pairs=True)
# # print(results)
# score = results[0][0]['qa-eval']['lerc_quip']

# a = [({'qa-eval': {'is_answered': 1.0, 'f1': 0.0, 'lerc_quip': 2.8713924884796143, 'em': 0.0}}, [[{'question': {
#     'question_id': 'dc50bcdddb09fd6e2772d349a1e8dd58', 'question': 'What is this?', 'answer': 'a summary',
#     'sent_start': 0, 'sent_end': 18, 'answer_start': 8, 'answer_end': 17}, 'prediction': {
#     'prediction_id': 'cfcd208495d565ef66e7dff9f98764da', 'prediction': 'a source document',
#     'probability': 0.7351705599072125, 'null_probability': 2.7699351270606014e-05, 'start': 8, 'end': 25,
#     'is_answered': 1.0, 'f1': 0, 'lerc_quip': 2.8713924884796143, 'em': 0}}]], [[{
#                                                                                      'question_id': 'dc50bcdddb09fd6e2772d349a1e8dd58',
#                                                                                      'question': 'What is this?',
#                                                                                      'answer': 'a summary',
#                                                                                      'sent_start': 0, 'sent_end': 18,
#                                                                                      'answer_start': 8,
#                                                                                      'answer_end': 17}]], [[{
#                                                                                                                 'prediction_id': 'cfcd208495d565ef66e7dff9f98764da',
#                                                                                                                 'prediction': 'a summary',
#                                                                                                                 'probability': 0.5786106138649608,
#                                                                                                                 'null_probability': 0.0019125114871777074,
#                                                                                                                 'start': 8,
#                                                                                                                 'end': 17}]])]

f = open('14model_outputs.jsonl', 'r')
new_f = open('qafacteval_14model_outputs.jsonl', 'w')
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
        results = metric.score_batch_qafacteval([article], [[model_sum]],
                                                return_qa_pairs=True)
        score = results[0][0]['qa-eval']['lerc_quip']
        new_model_outputs[key] = {'model_summ': model_sum, 'qafacteval_score': score}

    new_inst['model_output'] = new_model_outputs
    new_f.write(json.dumps(new_inst) + '\n')

new_f.close()