#!/bin/bash
folds='1 2 3 4 5'
for fold_num in $folds
do
    python3 train_classifier.py --fold_num $fold_num --init_lr 0.001 --lr_sched "cosS"
done