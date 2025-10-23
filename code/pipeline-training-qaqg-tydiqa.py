import json
from huggingface_hub import login
from dotenv import load_dotenv
import os
import argparse
import nltk, evaluate
from datasets import Dataset
import pyarrow as pa
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5Model
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from nltk import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)

import argparse
from sklearn.model_selection import train_test_split

# =================================================================================================================
# membuat data untuk answer extraction
# ==================================================================================================================

def get_context_answer_tydiqa(data):
    ''' 
    generate data for answer extraction with prefix-prepend method 
    context = prefix+context --> answers = item [SEP] item [SEP]
    '''
    data_qa = []
    for i, each in enumerate(data):
        answers = each['answer']
        context = 'extract answer: '+each['context']
        data_qa.append({'context': context, 'target': answers})
    return data_qa

def get_answer_baseline_tydiqa(data):
    """
    generate data for answer extraction with highlight method (BASELINE)
    context = context with <hl> answer <hl> --> answers = item [SEP]
    """
    data_qa = []
    for item in data:
        context = item['context']
        sentences = sent_tokenize(context)
        answers = item['answer']
        answers_idx_start = item['answer_start']
        answers_idx_end = item['answer_start']+len(answers) 
        sentence_idx_start = 0
        for i, sentence in enumerate(sentences):
            sentence_idx_end = sentence_idx_start + len(sentence)
            answer_list = []
            if answers_idx_start > sentence_idx_start and answers_idx_end < sentence_idx_end:
                sentences[i] = f"<hl> {sentence} <hl>"
            sentence_idx_start = sentence_idx_end
        sentences = " ".join(sentences)
        data_qa.append({'context': f"extract answer: {sentences}", 'target': answers })
    return data_qa

def get_sentence_answer_tydiqa(data):
    """
    generate data for answer extraction with prepend sentence method
    context = prefix+sentence containing answer --> answers = item [SEP]
    """
    data_qa = []
    for item in data:
        context = item['context']
        sentences = sent_tokenize(context)
        answers = item['answer']
        answers_idx_start = item['answer_start']
        answers_idx_end = item['answer_start']+len(answers) 
        sentence_idx_start = 0
        for i, sentence in enumerate(sentences):
            sentence_idx_end = sentence_idx_start + len(sentence)
            answer_list = []
            if answers_idx_start > sentence_idx_start and answers_idx_end < sentence_idx_end:
                target_sentence = sentence
            sentence_idx_start = sentence_idx_end
        data_qa.append({'context': f"extract answer: {target_sentence}", 'target': answers })
    return data_qa
    
# =================================================================================================================
# membuat data untuk question generation
# ==================================================================================================================

def get_context_question_tydiqa(data):
    """
    generate data for question generation with prefix-prepend method
    context = prefix+context [SEP] answer --> question = question
    """
    data_qg = []
    for i, each in enumerate(data):
        answer = each['answer']
        context = f"generate question: {each['context']} [SEP] {answer}"
        question = each['question']
            
        data_qg.append({'context': context, 'target': question})
    return data_qg

def get_question_baseline_tydiqa(data):
    """
    generate data for question generation with highlight method (BASELINE)
    context = context with <hl> answer <hl> --> question = question
    """
    data_qg = []
    for i, each in enumerate(data):
        answer = each['answer']
        context = each['context']
        answer_start = each['answer_start']
        context_new = f"generate question: {context[:answer_start]} <hl> {answer} <hl> {context[answer_start+len(answer):]}"
        question = each['question']
            
        data_qg.append({'context': context_new, 'target': question})
    return data_qg

def data_split(data_qa, data_qg, size, seed=42):
    """
    split the data into train and test set
    """
    qa_data = data_qa[:size]
    qg_data = data_qg[:size]
    print(len(data_qa), len(data_qg))

    qa_train, qa_test = train_test_split(qa_data, test_size=0.2, random_state=seed, shuffle=True)
    qg_train, qg_test = train_test_split(qg_data, test_size=0.2, random_state=seed, shuffle=True)

    train_set = qa_train + qg_train
    test_set = qa_test + qg_test

    train_set = Dataset.from_pandas(pd.DataFrame(list(train_set)))
    test_set = Dataset.from_pandas(pd.DataFrame(list(test_set)))

    print(train_set)
    print(test_set)

    return train_set, test_set


if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()

    # Automatically log in to Hugging Face using the retrieved token
    login(token=os.getenv("HF_LOGIN"))

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=10000)

    args = parser.parse_args()

    # define kode simpan
    kode_simpan = f'qaqg-v{args.scenario}.{args.seed}-TydiQA-id'

    # load the tydiqa preprocessed data
    with open('data/tydiqa-preprocesed.json', 'r') as f:
        data = json.load(f)

    # generate the data according to the scenario
    data_qa = []
    data_qg = []

    match args.scenario:
        case 1:
            data_qa = get_context_answer_tydiqa(data)
            data_qg = get_context_question_tydiqa(data)
        case 2:
            data_qa = get_sentence_answer_tydiqa(data)
            data_qg = get_context_question_tydiqa(data)
        case 0:
            data_qa = get_answer_baseline_tydiqa(data)
            data_qg = get_question_baseline_tydiqa(data)
    

    #only take this amount of data (squad is too big to train)
    qa_data = data_qa[:10000]
    qg_data = data_qg[:10000]

    # split the data
    train_set, test_set = data_split(qa_data, qg_data, size=args.size, seed=args.seed)

    # convert to Dataset
    train_set = Dataset.from_pandas(pd.DataFrame.from_dict(data=list(train_set), orient='columns'))
    test_set = Dataset.from_pandas(pd.DataFrame.from_dict(data=list(test_set), orient='columns'))

    print(train_set)
    print(test_set)

    # load the tokenizer and model
    model_checkpoint = 'muchad/idt5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

    # preprocessing the datasets
    max_input_length = 1024
    max_target_length = 128

    def preprocess_function(examples):
        inputs = [doc for doc in examples['context']]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples['target'], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # putting the data into mapping tokenized thing
    preprocessed_train_data = train_set.map(preprocess_function, batched=True)
    preprocessed_test_data = test_set.map(preprocess_function, batched=True)

    tokenized_datasets = {'train': preprocessed_train_data, 'test': preprocessed_test_data}

    # define metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    def compute_metrics(eval_preds):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids

        # decode preds and labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        
        return {
            'rouge1': rouge_result['rouge1'],
            'rouge2': rouge_result['rouge2'],
            'rougeL': rouge_result['rougeL'],
            'rougeLsum': rouge_result['rougeLsum'],
            "bleu": bleu_result["bleu"],
            'rouge_all': rouge_result,
            'bleu_all': bleu_result,
        }

    # define training arguments and trainer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    for param in model.parameters():
        param.data = param.data.contiguous()

    batch_size = 4
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-{kode_simpan}",
        overwrite_output_dir = True,
        eval_strategy = "epoch",
        save_strategy= "epoch", 
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=5,
        predict_with_generate=True,
        push_to_hub=True,
        load_best_model_at_end = False,
        use_cpu=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # train the model
    trainer.train()
    trainer.push_to_hub()
