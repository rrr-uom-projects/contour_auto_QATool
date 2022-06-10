#!/bin/bash
python3 test_classifier.py --init_lr 0.001 --lr_sched "cosS"
python3 test_classifier.py --init_lr 0.001 --lr_sched "cosS" --encAbl "True"
python3 test_classifier.py --init_lr 0.001 --lr_sched "cosS" --GNNAbl "True"
python3 test_classifier.py --init_lr 0.001 --lr_sched "cosS" --preAbl "True"