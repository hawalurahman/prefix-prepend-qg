#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install nltk -q')
get_ipython().system('pip install transformers -q')
get_ipython().system('pip install datasets -q')
get_ipython().system('pip install evaluate -q')
get_ipython().system('pip install rouge_score -q')
get_ipython().system('pip install sentencepiece -q')
get_ipython().system('pip install transformers[torch] -q')


# In[ ]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# In[ ]:


from datasets import Dataset
import pyarrow as pa
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5Model
import nltk, evaluate
from nltk import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)

def data_preparation(data):
    data_qa = []
    data_qg = []

    # membuat data untuk answer extraction
    # context = prefix+context --> answers = item [SEP] item [SEP]
    for i, each in enumerate(data):
        answers = ' [SEP] '.join(each['answers'])
        context = 'extract answer: '+each['context']
        data_qa.append({'context': context, 'target': answers})

    # membuat data untuk question generation
    # context = prefix+context [SEP] answer --> question = question
    for i, each in enumerate(data):
        for j, answer in enumerate(each['answers']):
            context = f"generate question: {each['context']} [SEP] {answer}"
            question = each['questions'][j]
            data_qg.append({'context': context, 'target': question})

    return data_qa, data_qg

def get_context_answers(cqa):
    '''Untuk Answer Extraction. mengambil context: prefix+paragraph --> output: answer [sep] answer [sep] .. answer'''
    hasil = []
    prefix = "extract answers: "
    for item in cqa:
        context = prefix+item['context']
        # answers = " [SEP] ".join([qa['answer'] for qa in item['qa']])
        answers = " [SEP] ".join([qa['answer'] for qa in item['qa']]) # usng default eos token for separator doesnt work
        hasil.append({'context': context, 'target': answers})

    return hasil

def get_context_question(cqa):
    hasil = []
    prefix = "generate questions: "

    for item in cqa:
        context = item['context']
        for qa in item['qa']:
            question = qa['question']
            answer = qa['answer']
            # hasil.append({'context': prefix+context+" [SEP] "+answer, 'target': question})
            hasil.append({'context': prefix+context+" [SEP] "+answer, 'target': question}) # using default eos token for separator

    return hasil

def get_context_question2(cqa):
    hasil = []
    prefix = "generate questions: "

    for item in cqa:
        context = item['context']
        for qa in item['qa']:
            question = qa['question']
            answer = qa['answer']
            # hasil.append({'context': prefix+context+" [SEP] "+answer, 'target': question})
            hasil.append({'context': prefix+context+" <answer> "+answer, 'target': question}) # using default eos token for separator

    return hasil

def get_sentence_answers(cqa):
    '''Untuk Answer Extraction. mengambil context: prefix + SENTENCE --> output: answer [sep] answer [sep] .. answer. 
    Setiap sentence dipastikan berada dalam kalimat yang mencakup idx pada start_answer dataset asli. 
    Sehingga tidak ada duplikasi jawaban muncul di tempat yang tidak seharusnya. '''

    prefix = "extract answers: "
    hasil = []
    for item in cqa:
        context = item['context']
        sentences = sent_tokenize(context)
        answers = [qa['answer'] for qa in item['qa']]
        answers_idx_start = [qa['answer_start'] for qa in item['qa']]
        answers_idx_end = [qa['answer_start']+len(qa['answer']) for qa in item['qa']]
        sentence_idx_start = 0
        for sentence in sentences:
            sentence_idx_end = sentence_idx_start + len(sentence)
            answer_list = []
            for i, answer in enumerate(answers):
                if answers_idx_start[i] > sentence_idx_start and answers_idx_end[i] < sentence_idx_end:
                    answer_list.append(answer)
            if len(answer_list) > 0:
                answer_list = " [SEP] ".join(answer_list)
                hasil.append({'context': prefix+sentence, 'target': answer_list})
                sentence_idx_start = sentence_idx_end
    return hasil

def get_context_hl_question(cqa):
    '''Highlight the sentence that contains the answer. However, not every occurences matter. 
    There should be a way to highlight specific word, append the paragraph with the sentence that highlighted'''

    prefix = "generate questions: "

    hasil = []
    for item in cqa:
        context = item['context']
        sentences = sent_tokenize(context)
        sentences_copy = sentences
        answers = [qa['answer'] for qa in item['qa']]
        question = [qa['question'] for qa in item['qa']]
        answers_idx_start = [qa['answer_start'] for qa in item['qa']]
        answers_idx_end = [qa['answer_start']+len(qa['answer']) for qa in item['qa']]
        sentence_idx_start = 0
        for j, sentence in enumerate(sentences):
            sentence_idx_end = sentence_idx_start + len(sentence)
            for i, answer in enumerate(answers):
                if answers_idx_start[i] > sentence_idx_start and answers_idx_end[i] < sentence_idx_end:
                    # print(sentence[:50], answer)
                    sentences_copy = sentences.copy()
                    temp = f" <hl> {sentence} <hl> "
                    sentences_copy[j] = temp
                    hasil.append({'context':prefix+"".join(sentences_copy)+" [SEP] "+answer, 'target': question[i]})
                else:
                    continue
            # if len(answer_list) > 0:
            #     answer_list = " [SEP] ".join(answer_list)
            #     hasil.append({'context': prefix+sentence, 'target': answer_list})
            sentence_idx_start = sentence_idx_end
    return hasil

def data_preparation_v2(data):
    '''idt5-base-qaqg'''
    data_qa = get_context_answers(data)
    data_qg = get_context_question(data)

    return data_qa, data_qg

def data_preparation_v3(data):
    '''idt5-base-qaqg-v2'''
    data_qa = get_sentence_answers(data)
    data_qg = get_context_question(data)

    return data_qa, data_qg

def data_preparation_v4(data):
    '''idt5-base-qaqg-v3'''
    data_qa = get_context_answers(data)
    data_qg = get_context_hl_question(data)

    return data_qa, data_qg

def data_preparation_v5(data):
    '''idt5-base-qaqg-v4'''
    data_qa = get_sentence_answers(data)
    data_qg = get_context_question2(data)

    return data_qa, data_qg

def data_split(data_qa, data_qg, size):  # split data 80:20 for training testing from single data with given size
    qa_data = data_qa[:size]
    qg_data = data_qg[:size]
    print(len(data_qa), len(data_qg))
    # Assuming 'data' is your dataset
    qa_train_set, qa_test_set = np.split(qa_data, [int(0.80 * len(qa_data))])
    qg_train_set, qg_test_set = np.split(qg_data, [int(0.80 * len(qg_data))])

    # Combine training sets
    train_set = np.concatenate((qa_train_set, qg_train_set))
    # Combine testing sets
    test_set = np.concatenate((qa_test_set, qg_test_set))

    train_set = Dataset.from_pandas(pd.DataFrame.from_dict(data=list(train_set), orient='columns'))
    test_set = Dataset.from_pandas(pd.DataFrame.from_dict(data=list(test_set), orient='columns'))

    print(train_set)
    print(test_set)

    return train_set, test_set

def init_tokenizer(model):
    tokenizer = T5Tokenizer.from_pretrained(model)

    return tokenizer

def preprocess_function(examples, ): #this is actually just tokenization using tokenizer
    max_input_length = 1024
    max_target_length = 128

    inputs = [doc for doc in examples['context']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples['target'], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def init_evaluation():
    nltk.download("punkt", quiet=True)
    nltk.download('punkt_tab', quiet=True)

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    return rouge, bleu

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute ROUGE scores
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Compute BLEU score
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)

    # Return both ROUGE and BLEU scores
    result = {
        'rouge1': rouge_result['rouge1'],
        'rouge2': rouge_result['rouge2'],
        'rougeL': rouge_result['rougeL'],
        'rougeLsum': rouge_result['rougeLsum'],
        "bleu": bleu_result["bleu"],  # Access the BLEU score from the result dictionary
    }
    return result

def init_model(model_checkpoint):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    for param in model.parameters():
        param.data = param.data.contiguous()

    return model


# In[ ]:


import json

with open('squad_id_possible.json') as f:
    data = json.load(f)


# In[ ]:


from huggingface_hub import login

# Automatically log in to Hugging Face using the retrieved token
login(token='')


# In[ ]:


model_checkpoint = 'muchad/idt5-base'
kode_simpan = 'qaqg_v1-1'

data = data
data_qa_qg = data_preparation_v2(data)
data_train_test = data_split(data_qa_qg[0], data_qa_qg[1], 30000)
tokenizer = init_tokenizer(model_checkpoint)
model = init_model(model_checkpoint)

preprocessed_train_data = data_train_test[0].map(preprocess_function, batched=True)
preprocessed_test_data = data_train_test[1].map(preprocess_function, batched=True)

tokenized_datasets = {'train': preprocessed_train_data, 'test': preprocessed_test_data}

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

rouge, bleu = init_evaluation()

batch_size = 8
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-{kode_simpan}",
    overwrite_output_dir = True,
    evaluation_strategy = "epoch",
    save_strategy= "epoch", 
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
#     weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=5,
    predict_with_generate=True,
    push_to_hub=False,
    load_best_model_at_end = True,
    use_cpu=False,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.push_to_hub()

