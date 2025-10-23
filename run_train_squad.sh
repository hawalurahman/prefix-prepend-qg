#!/bin/bash

python3 code/pipeline-training-qaqg.py --scenario 1 --seed 42 2>&1 | tee -a log_train_squad_v1_s42.txt
python3 code/pipeline-training-qaqg.py --scenario 1 --seed 12 2>&1 | tee -a log_train_squad_v1_s12.txt
python3 code/pipeline-training-qaqg.py --scenario 1 --seed 72 2>&1 | tee -a log_train_squad_v1_s72.txt

# python3 code/pipeline-training-qaqg.py --scenario 2 --seed 42 2>&1 | tee -a log_train_squad_v2_s42.txt
# python3 code/pipeline-training-qaqg.py --scenario 2 --seed 12 2>&1 | tee -a log_train_squad_v2_s12.txt
# python3 code/pipeline-training-qaqg.py --scenario 2 --seed 72 2>&1 | tee -a log_train_squad_v2_s72.txt