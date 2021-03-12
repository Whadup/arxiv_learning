# Datasets
## Finetuning Data
We provide three different fine-tuning datasets, each split into train and test files.

- __Equalities__ (=):  [Train](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_equalities_train.jsonl.gz) and [Test](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_equalities_test.jsonl.gz)
- __Inequalities__ (< and ≤ ): [Train](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_inequalities_train.jsonl.gz) and [Test](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_inequalities_test.jsonl.gz)
- __Mixed Operators__ (=<≤>≥): [Train](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_relations_train.jsonl.gz) and [Test](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_relations_test.jsonl.gz)

The fine-tuning data is based on the arxiv publications listed in the meta-datafile. When using the fine-tuning data in a downstream evaluation, make sure that your base-model is not trained on the same papers.
For sake of completeness, we have included the list of papers we used for pretraining.

- __Fine-Tuning Papers:__ [Metadata](https://github.com/Whadup/arxiv_learning/blob/master/data/test_papers_meta.json.gz)
- __Pre-Training Papers:__ [Metadata](https://github.com/Whadup/arxiv_learning/blob/master/data/train_papers_meta.json.gz)
