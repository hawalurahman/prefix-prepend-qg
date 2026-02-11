from datasets import Dataset
import json
import pyarrow as pa
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, T5Tokenizer, T5Model, pipeline
import nltk, evaluate
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to the training data')
parser.add_argument('--model_name', type=str, help='model name or path')
args = parser.parse_args()

kode_simpan = args.model_name.replace('hawalurahman/idt5-base-qaqg-', '').replace('.', '-').replace('/', '-')

# pipe = pipeline("text2text-generation", model=args.model_name, device_map='auto')

model = AutoModelForSeq2SeqLM.from_pretrained(
    args.model_name,
    local_files_only=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    local_files_only=True,
    use_fast=False
)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


def load_data(filepath):
    if filepath.endswith('.jsonl'):
        data = []
        with open(filepath) as f:
            for line in f:
                data.append(json.loads(line))
            data = [item for item in data if item['is_impossible'] == False]
        print("data loaded")
        return data
    else:
        with open(filepath) as f:
            data = json.load(f)
        print("data loaded")
    return data

def answer_extraction(context):
    inputs = f"extract answer: {context}"
    outputs = pipe(inputs)
    answer = outputs[0]['generated_text']
    return answer

def question_generation(context, answer):
    inputs = f"Generate question: {context} [SEP] {answer}"
    outputs = pipe(inputs)
    question = outputs[0]['generated_text']
    return question

data = load_data(args.data_path)
context_only = pd.DataFrame([item['context'] for item in data])
unique_context = list(context_only[0].unique())

from tqdm import tqdm 

generated_answers = [answer_extraction(item) for item in tqdm(unique_context)]

pd.DataFrame(generated_answers, columns=['generated_text']).to_json(f'answer-extraction-{kode_simpan}.jsonl', orient='records', lines=True)

generated_question = [question_generation(item['context'], item['answer'][0]) for item in tqdm(data)]

pd.DataFrame(generated_question, columns=['generated_text']).to_json(f'question-generation-{kode_simpan}.jsonl', orient='records', lines=True)



