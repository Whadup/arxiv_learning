"""
We train another model, facebooks fasttext, on the pre-order trees.
Then we test with finetuning tasks of equality prediction.
"""
import fasttext
import numpy as np
import json
import zipfile
import pickle
import meticulous

def train(dim=64, epoch=1, ws=5, ngrams=2):
    model = fasttext.train_unsupervised(
        "/data/s1/pfahler/arxiv_v2/plaintext_train.txt",
        model="skipgram",
        dim=dim,
        ws=ws,
        epoch=epoch,
        wordNgrams=ngrams,
    )
    return model

def main():
    config = dict(dim=64, epoch=1, ws=5, ngrams=2)
    with meticulous.Experiment(config):
        train(**config)

if __name__ == "__main__":
    main()