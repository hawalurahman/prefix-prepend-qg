#!/bin/bash

# python3 code/pipeline-training-qaqg.py --scenario 1 --seed 42 2>&1 | tee -a log_train_squad_v1_s42.txt
# python3 code/pipeline-training-qaqg.py --scenario 1 --seed 12 2>&1 | tee -a log_train_squad_v1_s12.txt
# python3 code/pipeline-training-qaqg.py --scenario 1 --seed 72 2>&1 | tee -a log_train_squad_v1_s72.txt

# python3 code/pipeline-training-qaqg.py --scenario 2 --seed 42 2>&1 | tee -a log_train_squad_v2_s42.txt
# python3 code/pipeline-training-qaqg.py --scenario 2 --seed 12 2>&1 | tee -a log_train_squad_v2_s12.txt
# python3 code/pipeline-training-qaqg.py --scenario 2 --seed 72 2>&1 | tee -a log_train_squad_v2_s72.txt

# python3 code/pipeline-training-qaqg-tydiqa.py --scenario 1 --seed 42 2>&1 | tee -a log_train_tydiqa_v1_s42.txt
# python3 code/pipeline-training-qaqg-tydiqa.py --scenario 1 --seed 12 2>&1 | tee -a log_train_tydiqa_v1_s12.txt
# python3 code/pipeline-training-qaqg-tydiqa.py --scenario 1 --seed 72 2>&1 | tee -a log_train_tydiqa_v1_s72.txt

# python3 code/pipeline-training-qaqg-tydiqa.py --scenario 2 --seed 42 2>&1 | tee -a log_train_tydiqa_v2_s42.txt
# python3 code/pipeline-training-qaqg-tydiqa.py --scenario 2 --seed 12 2>&1 | tee -a log_train_tydiqa_v2_s12.txt
# python3 code/pipeline-training-qaqg-tydiqa.py --scenario 2 --seed 72 2>&1 | tee -a log_train_tydiqa_v2_s72.txt

# FOR SQUAD ABLATION STUDIES

# python3 code/pipeline-training-qaqg.py --scenario 5 --seed 42 2>&1 | tee -a log_train_squad_v5_s42.txt
# python3 code/pipeline-training-qaqg.py --scenario 6 --seed 42 2>&1 | tee -a log_train_squad_v6_s42.txt
# python3 code/pipeline-training-qaqg.py --scenario 7 --seed 42 2>&1 | tee -a log_train_squad_v7_s42.txt
# python3 code/pipeline-training-qaqg.py --scenario 8 --seed 42 2>&1 | tee -a log_train_squad_v8_s42.txt
# python3 code/pipeline-training-qaqg.py --scenario 9 --seed 42 2>&1 | tee -a log_train_squad_v9_s42.txt
# python3 code/pipeline-training-qaqg.py --scenario 10 --seed 42 2>&1 | tee -a log_train_squad_v10_s42.txt
# python3 code/pipeline-training-qaqg.py --scenario 11 --seed 42 2>&1 | tee -a log_train_squad_v11_s42.txt
# python3 code/pipeline-training-qaqg.py --scenario 12 --seed 42 2>&1 | tee -a log_train_squad_v12_s42.txt


# FOR TYDIQA ABLATION STUDIES

CUDA_VISIBLE_DEVICES=0 python3 code/pipeline-training-qaqg-tydiqa.py --scenario 3 --seed 42 2>&1 | tee -a log_train_tydiqa_v3_s42.txt &
# python3 code/pipeline-training-qaqg-tydiqa.py --scenario 4 --seed 42 2>&1 | tee -a log_train_tydiqa_v4_s42.txt
CUDA_VISIBLE_DEVICES=1 python3 code/pipeline-training-qaqg-tydiqa.py --scenario 5 --seed 42 2>&1 | tee -a log_train_tydiqa_v5_s42.txt
# python3 code/pipeline-training-qaqg-tydiqa.py --scenario 6 --seed 42 2>&1 | tee -a log_train_tydiqa_v6_s42.txt
# python3 code/pipeline-training-qaqg-tydiqa.py --scenario 7 --seed 42 2>&1 | tee -a log_train_tydiqa_v7_s42.txt
# python3 code/pipeline-training-qaqg-tydiqa.py --scenario 8 --seed 42 2>&1 | tee -a log_train_tydiqa_v8_s42.txt
# python3 code/pipeline-training-qaqg-tydiqa.py --scenario 9 --seed 42 2>&1 | tee -a log_train_tydiqa_v9_s42.txt
