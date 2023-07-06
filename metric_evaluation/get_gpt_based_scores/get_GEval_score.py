import time
import sys
import openai
import os
from tqdm import tqdm
import json

# openai.api_base = 'https://api.openai-asia.com/v1'
# openai.api_key = 'sk-437Tr7gfq8RXS7cnQQ6Vky5pXeHFqAYpWaBxLNAjbD17qcVU'
# # "https://closeai.deno.dev/v1"

openai.api_base = 'https://openai.api2d.net/v1'
openai.api_key = 'fk205563-yEH2DxljhJV1dV1qa518UGTgIeKqYl4H'


class GPT3Model(object):

    def __init__(self, model_name, api_key):
        self.model_name = model_name
        # try:
        #     openai.api_key = api_key
        # except Exception:
        #     pass
        self.dimensions = ['Aspect Relevance', 'Self-Coherence', 'Sentiment Consistency', 'Readability']
        f = open('geval_instructions_mod.jsonl')
        self.CoT = json.loads(f.readline())

    def get_GEval_score(self):
        processed_cases = 0
        while processed_cases < 100:
            try:
                # ------ data preparation ------
                if os.path.exists('text_ada_001_geval_score_mod.jsonl'):
                    f = open('text_ada_001_geval_score_mod.jsonl',
                             'r')  # in case the connection is lost, and you need to run it more than once
                    processed_cases = len(f.readlines())
                    # print(processed_cases)
                    f.close()

                model_f = open('../14model_outputs.jsonl', 'r')

                f = open('text_ada_001_geval_score_mod.jsonl', 'a')
                print('processed cases: {}'.format(processed_cases))

                for i, line in tqdm(enumerate(model_f)):
                    if i < processed_cases:
                        continue
                    inst = json.loads(line)
                    case = inst['case']
                    model_names = list(inst['model_output'].keys())
                    model_names.sort()
                    assert len(model_names) == 14
                    input_reviews = 'Reviews:\n1.{}\n2.{}\n3.{}\n4.{}\n5.{}\n6.{}\n7.{}\n8.{}\n'.format(
                        inst['revs']['rev1'],
                        inst['revs']['rev2'],
                        inst['revs']['rev3'],
                        inst['revs']['rev4'],
                        inst['revs']['rev5'],
                        inst['revs']['rev6'],
                        inst['revs']['rev7'],
                        inst['revs']['rev8'])
                    gpt_reply = {model_name: {dim: None for dim in self.dimensions} for model_name in model_names}
                    for model_idx in range(14):
                        input_summs = 'Summary: {}\n'.format(inst['model_output'][model_names[model_idx]])
                        for dim in self.dimensions:
                            instruction = self.CoT[dim]
                            prompt = instruction + '\n' + input_reviews + '\n' + input_summs
                            # print('processing summ {} dim {}'.format(model_idx+1, dim))
                            score_log_prob_dict = self.get_weighted_score(raw_prompt=prompt, dimension=dim)

                            gpt_reply[model_names[model_idx]][dim] = score_log_prob_dict

                    inst_dict = {'gpt_reply': gpt_reply, 'case': int(i + 1)}
                    f.write(json.dumps(inst_dict) + '\n')
                    print('instance {} is written'.format(i+1))
                    # if i == 0:
                    #     print('case 1 is finished')
                    #     return 0
                    time.sleep(20)

                f.close()
                processed_cases = case  # should be 100 to end the while loop; if < 100, the program will continue to try
                break
            except Exception:
                time.sleep(3)

    def get_weighted_score(self, raw_prompt, dimension):
        score_log_prob_dict = {}
        for score in [1, 2, 3, 4, 5]:
            # print('processing score: {}'.format(score))
            score_log_prob = self.do_inference(raw_prompt=raw_prompt, dimension=dimension, score=score)
            score_log_prob_dict[score] = score_log_prob
        return score_log_prob_dict

    def do_inference(self, raw_prompt, dimension, score):
        score_prompt = 'the score for \"{}\" is {}'.format(dimension, score)
        prompt = raw_prompt + score_prompt
        response = self.gpt3(prompt)
        out = response["choices"][0]
        # print(out)

        assert prompt == out["text"]

        # score_log_prob = out['logprobs']["token_logprobs"][-1]
        # assert str(score) in out['logprobs']["tokens"][-1]

        # return score_log_prob
        return out

    def gpt3(self, prompt, max_len=0, temp=0, num_log_probs=0, echo=True, n=None):
        response = None
        received = False
        while not received:
            try:
                # print('getting response...')
                response = openai.Completion.create(model=self.model_name,
                                                    prompt=prompt,
                                                    max_tokens=max_len,
                                                    # setting this to 0 guarantees that the GPT model will not generate anything
                                                    temperature=temp,
                                                    logprobs=num_log_probs,
                                                    echo=echo,
                                                    stop='\n',
                                                    n=n)
                # print('prompt: ', prompt)
                received = True
                # print('received')
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response


if __name__ == "__main__":
    M = GPT3Model(model_name='text-ada-001', api_key=None)
    M.get_GEval_score()
