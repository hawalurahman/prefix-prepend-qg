#!/bin/bash

# First batch (run 3 at once)
# python3 code/inference.py --data_path data/squad-dev.jsonl --model_name hawalurahman/idt5-base-qaqg-v1.42-SQuAD-id &
# python3 code/inference.py --data_path data/squad-dev.jsonl --model_name hawalurahman/idt5-base-qaqg-v1.12-SQuAD-id &
# python3 code/inference.py --data_path data/squad-dev.jsonl --model_name hawalurahman/idt5-base-qaqg-v1.72-SQuAD-id &
# wait

# Second batch
# python3 code/inference.py --data_path data/squad-dev.jsonl --model_name hawalurahman/idt5-base-qaqg-v2.42-SQuAD-id &
# python3 code/inference.py --data_path data/squad-dev.jsonl --model_name hawalurahman/idt5-base-qaqg-v2.12-SQuAD-id &
# python3 code/inference.py --data_path data/squad-dev.jsonl --model_name hawalurahman/idt5-base-qaqg-v2.72-SQuAD-id &
# wait

# Third batch
# python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name hawalurahman/idt5-base-qaqg-v1.42-TydiQA-id &
# python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name hawalurahman/idt5-base-qaqg-v1.12-TydiQA-id &
# python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name hawalurahman/idt5-base-qaqg-v1.72-TydiQA-id &
# wait

# Fourth batch
# python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name hawalurahman/idt5-base-qaqg-v2.42-TydiQA-id &
# python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name hawalurahman/idt5-base-qaqg-v2.12-TydiQA-id &
# python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name hawalurahman/idt5-base-qaqg-v2.72-TydiQA-id &
# wait

# ABLATION STUDIES

# First batch (run 3 at once)
# python3 code/inference.py --data_path data/squad-dev.jsonl --model_name idt5-base-qaqg-ae-noprefix-noprepend.42-SQuAD-id/checkpoint-8475 &
# python3 code/inference.py --data_path data/squad-dev.jsonl --model_name idt5-base-qaqg-ae-yesprefix-noprepend.42-SQuAD-id/checkpoint-8475 &
# python3 code/inference.py --data_path data/squad-dev.jsonl --model_name idt5-base-qaqg-ae-yesprefix-yesprepend.42-SQuAD-id/checkpoint-8475 &
# wait

# Second batch
python3 code/inference.py --data_path data/squad-dev.jsonl --model_name idt5-base-qaqg-noprefix-noprepend.42-SQuAD-id/checkpoint-23475 &
python3 code/inference.py --data_path data/squad-dev.jsonl --model_name idt5-base-qaqg-qg-noprefix-noprepend.42-SQuAD-id/checkpoint-15000 &
python3 code/inference.py --data_path data/squad-dev.jsonl --model_name idt5-base-qaqg-qg-yesprefix-noprepend.42-SQuAD-id/checkpoint-15000 &
wait

# Third batch
python3 code/inference.py --data_path data/squad-dev.jsonl --model_name idt5-base-qaqg-qg-yesprefix-yesprepend.42-SQuAD-id/checkpoint-15000 &
python3 code/inference.py --data_path data/squad-dev.jsonl --model_name idt5-base-qaqg-yesprefix-noprepend.42-SQuAD-id/checkpoint-23475 &
wait

# Fourth batch
python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name idt5-base-qaqg-ae-noprefix-noprepend.42-TydiQA-id/checkpoint-5705 &
python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name idt5-base-qaqg-ae-yesprefix-noprepend.42-TydiQA-id/checkpoint-5705 &
python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name idt5-base-qaqg-noprefix-noprepend.42-TydiQA-id/checkpoint-5705 &
wait

# Fifth batch
python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name idt5-base-qaqg-qg-noprefix-noprepend.42-TydiQA-id/checkpoint-2855 &
python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name idt5-base-qaqg-qg-yesprefix-noprepend.42-TydiQA-id/checkpoint-2855 &
python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name idt5-base-qaqg-qg-yesprefix-yesprepend.42-TydiQA-id/checkpoint-2855 &
wait

# Sixth batch
python3 code/inference.py --data_path data/tydiqa-preprocesed-eval.json --model_name idt5-base-qaqg-yesprefix-noprepend.42-TydiQA-id/checkpoint-5705 &
wait