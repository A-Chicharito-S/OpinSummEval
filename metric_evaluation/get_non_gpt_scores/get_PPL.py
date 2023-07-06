from tqdm import tqdm
import torch
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast


def GPT2_ppl(text, model, tokenizer):

    encodings = tokenizer(text, return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512

    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.cpu().numpy().tolist()


def Pegasus_ppl(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors="pt")

    max_length = model.config.max_position_embeddings
    stride = 512

    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.cpu().numpy().tolist()


f = open('14model_outputs.jsonl', 'r')
new_f = open('ppl_14model_outputs.jsonl', 'w')

device = "cuda:0"

GPTmodel_id = "gpt2-large"
GPTmodel = GPT2LMHeadModel.from_pretrained(GPTmodel_id).to(device)
GPTtokenizer = GPT2TokenizerFast.from_pretrained(GPTmodel_id)

Pegasus_model_id = "google/pegasus-xsum"
Pegasus_model = PegasusForConditionalGeneration.from_pretrained(Pegasus_model_id).to(device)
Pegasus_tokenizer = PegasusTokenizerFast.from_pretrained(Pegasus_model_id)

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
        gpt_ppl = GPT2_ppl(text=hypo, model=GPTmodel, tokenizer=GPTtokenizer)
        pegasus_ppl = Pegasus_ppl(text=hypo, model=Pegasus_model, tokenizer=Pegasus_tokenizer)
        new_model_outputs[key] = {'model_summ': hypo, 'ppl_score': {'gpt2': gpt_ppl, 'pegasus': pegasus_ppl}}

    new_inst['model_output'] = new_model_outputs
    new_f.write(json.dumps(new_inst) + '\n')

new_f.close()
