import json
from tqdm import tqdm
import os
import openpyxl
import pandas as pd


class Sorter:
    def __init__(self, auto_file_path='autometric_scores', human_file_path='final_batch'):
        print('root path for automatic metrics: {}'.format(auto_file_path))
        print('root path for humann annotations: {}'.format(human_file_path))
        self.auto_file_path = auto_file_path
        self.human_file_path = human_file_path
        self.auto_file_list = os.listdir(auto_file_path)
        self.human_file_list = os.listdir(human_file_path)
        self.model_names = None

    def sort_json_file(self, file_path, auto_metric=True):
        root_path = self.auto_file_path if auto_metric else self.human_file_path
        file_list = os.listdir(root_path + '/' + file_path)
        if auto_metric:
            print('sorting automatic metrics...')
            assert len(file_list) == 1
            json_file_path = root_path + '/' + file_path + '/' + file_list[0]
            auto_f = open(json_file_path, 'r')
            overall_dict = {}
            print(json_file_path)
            for i, line in enumerate(tqdm(auto_f)):
                inst = json.loads(line)
                if 'case' in inst.keys():
                    case_num = str(inst['case'])
                else:
                    case_num = str(i + 1)

                if file_path == 'supert':
                    scores = inst['supert_score']
                    one_inst_scores = {}
                    for mdl_name in self.model_names:
                        one_inst_scores[mdl_name] = {'supert_score': scores[mdl_name]}
                    overall_dict[case_num] = one_inst_scores
                else:
                    model_outputs = inst['model_output']
                    model_names = list(model_outputs.keys())
                    model_names.sort()  # TODO: very important to keep the order here!
                    self.model_names = model_names
                    one_inst_scores = {}

                    for mdl_name in model_names:
                        one_model_scores = {}
                        model_infor = model_outputs[mdl_name]
                        # {'model_summ': xxx, 'metric-name1': {'xxx': xxx, ...}, 'metric-name2': {'xxx': xxx, ...}, ...}
                        metric_names = list(filter(lambda x: 'model' not in x, model_infor.keys()))

                        for metric_name in metric_names:
                            if isinstance(model_infor[metric_name], dict):
                                # print('multiple metrics: {}'.format(metric_name))
                                scores = model_infor[metric_name]
                                for score_name in scores.keys():
                                    if 'NLGEval' in metric_name or 'R1_F1' in score_name or 'R2_F1' in score_name or 'RL_F1' in score_name or 'bary_score' in metric_name:
                                        new_score_name = score_name
                                    else:
                                        new_score_name = metric_name + '_' + score_name
                                    if 'infoLM_score' in metric_name:
                                        sub_names = list(scores['ab_div'].keys())
                                        for sub_score_name in sub_names:
                                            one_model_scores[metric_name+'_'+sub_score_name] = float(scores['ab_div'][sub_score_name][0])
                                    elif 'bary_score' in metric_name:
                                        one_model_scores[new_score_name] = float(scores[score_name][0])
                                    else:
                                        if score_name != 'EmbeddingAverageCosineSimilairty' and score_name != 'ROUGE_L':
                                            # skip these two from NLGEval
                                            one_model_scores[new_score_name] = float(scores[score_name]) # if 'ppl' not in new_score_name else -float(scores[score_name])
                                            # add '-' to ppl scores
                            else:
                                # print('single metric: {}'.format(metric_name))
                                one_model_scores[metric_name] = model_infor[metric_name]

                        one_inst_scores[mdl_name] = one_model_scores

                    overall_dict[case_num] = one_inst_scores
                # {'1': {'bart': {'R1_F1': xxx, 'R2_F1': xxx, ...},
                #        't5': {'R1_F1': xxx, 'R2_F1': xxx, ...},
                #         ...
                #        },
                #  '2': {'some-content-similar-to-above-here'},
                #        ...
                #  }
            auto_f.close()

        else:
            print('dealing with human annotations...')
            assert self.model_names is not None
            overall_dict = {}
            eval_dimension = ['asp_rel', 'self_coh', 'sent_con', 'readblty']
            for batch_num in range(10):
                batch_file_path = root_path + '/' + file_path + '/' + list(filter(lambda x: 'batch' + str(batch_num + 1) in x, file_list))[0]
                human_sheets = openpyxl.load_workbook(batch_file_path)
                for case_num in range(10):
                    one_inst_scores = {}
                    sheet_name = 'case' + str(batch_num * 10 + case_num + 1)
                    one_case_sheet = human_sheets[sheet_name]
                    for mdl_idx, row in enumerate(one_case_sheet['13:26']):
                        one_model_scores = {}
                        for eval_idx, cell in enumerate(row[:4]):
                            # dimension_name = file_path + '_' + eval_dimension[eval_idx]
                            dimension_name = eval_dimension[eval_idx]
                            # file_path here is either 'annot1' or 'annot2'
                            one_model_scores[dimension_name] = float(cell.value)
                        mdl_name = self.model_names[mdl_idx]
                        one_inst_scores[mdl_name] = {file_path: one_model_scores}
                    overall_dict[str(batch_num * 10 + case_num + 1)] = one_inst_scores
        assert len(overall_dict) == 100, 'the file is: '.format(file_path)
        return overall_dict

    def sort_out_file(self, file_path):
        # specifically for sms
        root_path = self.auto_file_path
        file_list = os.listdir(root_path + '/' + file_path)
        overall_dict = {}
        for path in file_list:
            sms_file_path = root_path + '/' + file_path + '/' + path
            df = pd.read_csv(sms_file_path, sep='\t').iloc[1:]
            if 'glove' in path:
                # deals with glove embeddings
                print('glove embeddings')
                assert len(df) == 1400
                for row in df.iterrows():
                    idx = int(row[0][0])
                    # model_summ = row[0][2]
                    score = row[1].tolist()[0]
                    case_num = str(int(idx / 14) + 1)
                    model_num = idx % 14
                    if case_num not in overall_dict.keys():
                        overall_dict[case_num] = {}
                    if self.model_names[model_num] not in overall_dict[case_num].keys():
                        overall_dict[case_num][self.model_names[model_num]] = {}
                    overall_dict[case_num][self.model_names[model_num]]['sms_glove'] = float(score)
            else:
                print('elmo embeddings')
                assert len(df) == 1400
                for row in df.iterrows():
                    idx = int(row[0][0])
                    # model_summ = row[0][2]
                    score = row[1].tolist()[0]
                    case_num = str(int(idx / 14) + 1)
                    model_num = idx % 14
                    if case_num not in overall_dict.keys():
                        overall_dict[case_num] = {}
                    if self.model_names[model_num] not in overall_dict[case_num].keys():
                        overall_dict[case_num][self.model_names[model_num]] = {}
                    overall_dict[case_num][self.model_names[model_num]]['sms_elmo'] = float(score)
        assert len(overall_dict) == 100, 'the file is: '.format(file_path)
        return overall_dict
        # {'1': {'bart': {'R1_F1': xxx, 'R2_F1': xxx, ...},
        #        't5': {'R1_F1': xxx, 'R2_F1': xxx, ...},
        #         ...
        #        },
        #  '2': {'some-content-similar-to-above-here'},
        #        ...
        #  }

    def sort_csv_file(self, file_path):
        root_path = self.auto_file_path
        file_list = os.listdir(root_path + '/' + file_path)
        assert len(file_list) == 1
        chrF_file_path = root_path + '/' + file_path + '/' + file_list[0]
        df = pd.read_csv(chrF_file_path).iloc[:, [0, 3]]
        overall_dict = {}
        assert len(df) == 1400
        for row in df.iterrows():
            idx = row[0]
            score = row[1].to_dict()['chrF_score']
            case_num = str(int(idx / 14) + 1)
            model_num = idx % 14
            if case_num not in overall_dict.keys():
                overall_dict[case_num] = {}
            if self.model_names[model_num] not in overall_dict[case_num].keys():
                overall_dict[case_num][self.model_names[model_num]] = {}
            overall_dict[case_num][self.model_names[model_num]]['chrF'] = score
        assert len(overall_dict) == 100, 'the file is: '.format(file_path)
        return overall_dict
        # {'1': {'bart': {'R1_F1': xxx, 'R2_F1': xxx, ...},
        #        't5': {'R1_F1': xxx, 'R2_F1': xxx, ...},
        #         ...
        #        },
        #  '2': {'some-content-similar-to-above-here'},
        #        ...
        #  }

    def create_score_json_file(self):
        dict_list = []
        for auto_path in self.auto_file_list:
            if auto_path == 'sms':
                dict_list.append(self.sort_out_file(file_path=auto_path))
            elif auto_path == 'chrF':
                dict_list.append(self.sort_csv_file(file_path=auto_path))
            else:
                dict_list.append(self.sort_json_file(file_path=auto_path, auto_metric=True))

        for human_path in self.human_file_list:
            dict_list.append(self.sort_json_file(file_path=human_path, auto_metric=False))
        print('putting all scores into one file...')
        f = open('scores.jsonl', 'w')
        for i in tqdm(range(100)):
            one_inst_dict = {model_name: {} for model_name in self.model_names}
            one_inst_dict['case'] = i + 1
            for metric_dict in dict_list:  # num of metric times
                one_inst_metric_dict = metric_dict[str(i + 1)]
                for model_name in self.model_names:  # 14 times
                    one_inst_dict[model_name].update(one_inst_metric_dict[model_name])
            f.write(json.dumps(one_inst_dict) + '\n')

        return 0


sorter = Sorter()
sorter.create_score_json_file()
