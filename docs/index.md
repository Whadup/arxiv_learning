## Arxiv Learning
We propose to use unsupervised representation learning techniques to search for relatex mathematical expressions on arxiv.org. A demo is running at:  
[https://heureka2.azurewebsites.net](https://heureka2.azurewebsites.net).

## Preprocessing

We have published our data processing pipeline as a standalone python library at [https://github.com/Whadup/arxiv_library](https://github.com/Whadup/arxiv_library). 
{% comment %} 
## Models

Lorem Ipsum

## Results

Lorem Ipsum
{% endcomment %}

## Datasets

### Preprocessed Data

Password protected, but available here: [Sciebo-Link](https://tu-dortmund.sciebo.de/s/Ul9xSFnbeQFV9qC). 
Just reach out for the password.

### Keyword-Annotated Formulas

in this shared LaTeX document, we collect keyword-annotated formulas: [Overleaf](https://www.overleaf.com/8721648589nrjxgwmtzfvm). We can query these formulas in a large collection of papers and check if the keywords appear in the context of the search results. A processed version of this document is available here: [eval.json](https://github.com/Whadup/arxiv_learning/blob/master/eval.json).

We also have a small set of machine-learning related formulas labeled into categories. [Labeled Data](https://github.com/Whadup/arxiv_learning/blob/master/eval.json). __Warning:__ These formulas are copied from arxiv papers, often multiple formulas belong to the same paper and we do not have meta data to reconstruct the source. When pre-training on a large collection, it is thus likely that these test formulas have been seen during training, possibly even as positive pairs in contrasitve learning tasks.

### Finetuning Data
We have automatically identified equalities and inequalities on arXiv. Now the machine learning task is to learn to match left-hand-sides and right-hand-sides of these (in-)equalities. We provide three different fine-tuning datasets, each split into train and test files. See [finetune_model.py#L72](https://github.com/Whadup/arxiv_learning/blob/799cb2f861fef70fdd46acf23c3ed97064655ef4/arxiv_learning/jobs/finetune_model.py#L72) for an example of how to evaluate the model with these datasets.

- __Equalities__ (=):  [Train](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_equalities_train.jsonl.gz) and [Test](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_equalities_test.jsonl.gz)
- __Inequalities__ (< and ≤ ): [Train](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_inequalities_train.jsonl.gz) and [Test](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_inequalities_test.jsonl.gz)
- __Mixed Operators__ (=<≤>≥): [Train](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_relations_train.jsonl.gz) and [Test](https://github.com/Whadup/arxiv_learning/blob/master/data/finetune_relations_test.jsonl.gz)

The fine-tuning data is based on the arxiv publications listed in the meta-datafile. When using the fine-tuning data in a downstream evaluation, make sure that your base-model is not trained on the same papers.
For sake of completeness, we have included the list of papers we used for pretraining.

- __Fine-Tuning Papers:__ [Metadata](https://github.com/Whadup/arxiv_learning/blob/master/data/test_papers_meta.json.gz)
- __Pre-Training Papers:__ [Metadata](https://github.com/Whadup/arxiv_learning/blob/master/data/train_papers_meta.json.gz)

## Finetuning Results

We report finetuning results for different kinds of models, measuring recall@K.

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Equalities</th>
    <th colspan="3">Inequalities</th>
    <th colspan="3">Mixed Operators</th>
  </tr>
  <tr>
    <td>R@1</td>
    <td>R@10</td>
    <td>R@100</td>
    <td>R@1</td>
    <td>R@10</td>
    <td>R@100</td>
    <td>R@1</td>
    <td>R@10</td>
    <td>R@100</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>FastText</td>
    <td>0.46</td>
    <td>0.64</td>
    <td>0.73</td>
    <td>0.48</td>
    <td>0.70</td>
    <td>0.80</td>
    <td>0.47</td>
    <td>0.63</td>
    <td>0.73</td>
  </tr>
  <tr>
    <td>Tangent-CFT</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>GraphCNN</td>
    <td>0.51</td>
    <td>0.83</td>
    <td>0.88</td>
    <td>0.50</td>
    <td>0.87</td>
    <td>0.92</td>
    <td>0.51</td>
    <td>0.83</td>
    <td>0.88</td>
  </tr>
  <tr>
    <td>Transformer small</td>
    <td>0.51</td>
    <td>0.74</td>
    <td>0.87</td>
    <td>0.50</td>
    <td>0.82</td>
    <td>0.96</td>
    <td>0.52</td>
    <td>0.74</td>
    <td>0.86</td>
  </tr>
</tbody>
</table>

## References

- Lukas Pfahler and Katharina Morik. "Semantic Search in Millions of Equations", *Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.* 2020. [Paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403056)
- Stefan Todorinski, "Erkennung von Ähnlichkeiten zwischen mathematischen Ausdrücken mittels Bidirectional Encoder Representations from Transformers", *Master Thesis*, Dortmund, 2021.
- Jonathan Schill, "Scaling up the Equation-Encoder - Handling High Data Volume through the Efficient Use of Trainable Parameters", *Bachelor Thesis*, Dortmund, 2020. [Paper](https://www-ai.cs.tu-dortmund.de/auto?self=%24g2mkm58yyo)
- Lukas Pfahler, Jonathan Schill, and Katharina Morik. "The Search for Equations–Learning to Identify Similarities between Mathematical Expressions." *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*. Springer, Cham, 2019. [Paper](https://link.springer.com/chapter/10.1007/978-3-030-46133-1_42)

