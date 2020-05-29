#!/usr/bin/env bash
for i in {1..10} 
do 
    python3 -u -m arxiv_learning.jobs.train_model  | grep "FINISHED_RUN" >> full_model.txt
    python3 -u -m arxiv_learning.jobs.train_model with masked_language_training=False | grep "FINISHED_RUN" >> without_lm_model.txt
    python3 -u -m arxiv_learning.jobs.train_model with data_augmentation=False | grep "FINISHED_RUN" >> without_augmentation_model.txt
done