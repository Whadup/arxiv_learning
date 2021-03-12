## Arxiv Learning
We propose to use unsupervised representation learning techniques to search for relatex mathematical expressions on arxiv.org. A demo is running at [https://heureka2.azurewebsites.net](https://heureka2.azurewebsites.net).

## Preprocessing

We have published our data processing pipeline as a standalone python library at [https://github.com/Whadup/arxiv_library](https://github.com/Whadup/arxiv_library). 
{% comment %} 
## Models

Lorem Ipsum

## Results

Lorem Ipsum
{% endcomment %}

## Datasets

### Keyword-Annotated Formulas

in this shared LaTeX document, we collect keyword-annotated formulas: https://www.overleaf.com/8721648589nrjxgwmtzfvm. We can query these formulas in a large collection of papers and check if the keywords appear in the context of the search results.

### Finetuning Data
We have automatically identified equalities and inequalities on arXiv. Now the machine learning task is to learn to match left-hand-sides and right-hand-sides of these (in-)equalities. We provide three different fine-tuning datasets, each split into train and test files. 

- __Equalities__ (=):  [Train](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_equalities_train.jsonl.gz) and [Test](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_equalities_test.jsonl.gz)
- __Inequalities__ (< and ≤ ): [Train](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_inequalities_train.jsonl.gz) and [Test](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_inequalities_test.jsonl.gz)
- __Mixed Operators__ (=<≤>≥): [Train](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_relations_train.jsonl.gz) and [Test](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_relations_test.jsonl.gz)

The fine-tuning data is based on the arxiv publications listed in the meta-datafile. When using the fine-tuning data in a downstream evaluation, make sure that your base-model is not trained on the same papers.
For sake of completeness, we have included the list of papers we used for pretraining.

- __Fine-Tuning Papers:__ [Metadata](https://github.com/Whadup/arxiv_learning/blob/master/data/test_papers_meta.json.gz)
- __Pre-Training Papers:__ [Metadata](https://github.com/Whadup/arxiv_learning/blob/master/data/train_papers_meta.json.gz)


## References

- Lukas Pfahler and Katharina Morik. "Semantic Search in Millions of Equations", *Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.* 2020. [Paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403056)
- Stefan Todorinski, "Erkennung von Ähnlichkeiten zwischen mathematischen Ausdrücken mittels Bidirectional Encoder Representations from Transformers", *Master Thesis*, Dortmund, 2020.
- Jonathan Schill, "Scaling up the Equation-Encoder - Handling High Data Volume through the Efficient Use of Trainable Parameters", *Bachelor Thesis*, Dortmund, 2020. [Paper](https://www-ai.cs.tu-dortmund.de/auto?self=%24g2mkm58yyo)
- Lukas Pfahler, Jonathan Schill, and Katharina Morik. "The Search for Equations–Learning to Identify Similarities between Mathematical Expressions." *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*. Springer, Cham, 2019. [Paper](https://link.springer.com/chapter/10.1007/978-3-030-46133-1_42)

