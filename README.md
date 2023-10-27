# OpinSummEval
``model_training`` contains code that helps to obtain the outputs from models: Opinosis, LexRank, text-davinci-003, and PLMs (T5, PEGASUS, BART). We provide the training/validation/test data for PLMs at:

**train**: https://drive.google.com/file/d/1wHEoPuLX15hJn12AIbkvDImqzfE1g77Z/view?usp=sharing

**validation**: https://drive.google.com/file/d/1ZhAhy0yF_dr_0-e1ibuHcH2UA75_Plfe/view?usp=sharing

**test**: https://drive.google.com/file/d/19osVsUanQpfgqwMLWEtB6PO3kOuv3WyW/view?usp=sharing

(the validation and test splits are from **MeanSum** [[paper]](https://proceedings.mlr.press/v97/chu19b/chu19b.pdf) [[code]](https://github.com/sosuperic/MeanSum) and some parts of the training code for PLMs are borrowed from **AceSum** [[paper]](https://aclanthology.org/2021.emnlp-main.528.pdf) [[code]](https://github.com/rktamplayo/AceSum). The codes for **Opinosis** and **LexRank** are directly from their official repo, which are specified in the appendix of our paper)

``metric_evaluation`` contains the code and data that help to get the scores from different automatic metrics, calculate correlations, etc.
