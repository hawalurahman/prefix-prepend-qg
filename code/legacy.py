import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding, DataCollatorForSeq2Seq, T5Tokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import argparse
import torch
from datasets import load_dataset, Dataset
import nltk
import evaluate
import numpy as np
from huggingface_hub import login

nltk.download("punkt")
nltk.download('punkt_tab')

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# Automatically log in to Hugging Face using the retrieved token
login(token='')

def main():
    parser = argparse.ArgumentParser(description="Train a model for MCQ generation.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model')
    parser.add_argument('--kode_simpan', type=str, required=True, help='Kode simpan for the model')
    parser.add_argument('--folder_dataset')

    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    # following old code
    if args.model_name == "muchad/idt5-base":
        for param in model.parameters():
            param.data = param.data.contiguous()
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # Load the training data
    ds = load_dataset(args.folder_dataset)

    def data_preparation(data):
        df = pd.DataFrame(data)
        grouped_df = df.groupby('context').agg({
            'context': 'first',
            'question': list,
            'answer': list,
        })

        answer_extraction = [{
            'input': f"Extract answers. Context: {row['context']}",
            'output': "".join([ans + " [SEP] " for item in row['answer'] for ans in item])
        } for i, row in grouped_df.iterrows()]
        
        print(answer_extraction[0])

        question_generation = [{'input': f'Generate question. Answer: {row['answer'][0].rstrip('.')}. Context: {row['context']}', 
                              'output': row['question']} 
                              for i, row in df.iterrows()]
        
        print(question_generation[0])

        question_answering = [{'input': f'Answer question. Context: {row['context']}. Question: {row['question']}. ', 
                              'output': row['answer'][0].rstrip('.')} 
                              for i, row in df.iterrows()]
        
        print(question_answering[0])
        
        
        return Dataset.from_list(answer_extraction + question_generation + question_answering)
    
    def preprocess_function(examples): #this is actually just tokenization using tokenizer
        max_input_length = 1024
        max_target_length = 128
        
        inputs = [doc for doc in examples['input']]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples['output'], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

    # compute metrics lama
    def compute_metrics(eval_preds): 
        print("EVAL PREDS")
        print(eval_preds)
        
        preds, labels = eval_preds
        print("PREDS")
        print(preds)
        print("LABELS")
        print(labels)

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
    
    # compute metrics baru
    def get_compute_metrics(tokenizer):
        def compute_metrics(eval_preds):
            preds = eval_preds.predictions
            labels = eval_preds.label_ids
            
            # preds, labels = eval_preds
            
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
            }
        return compute_metrics

    
    split = ds['train'].train_test_split(test_size=0.1, seed=42)
    train_data = data_preparation(split['train']).map(preprocess_function, batched=True)
    eval_data = data_preparation(split['test']).map(preprocess_function, batched=True)

    # debugging
    # train_data = data_preparation(split['train'].select(range(10))).map(preprocess_function, batched=True)
    # eval_data = data_preparation(split['test'].select(range(10))).map(preprocess_function, batched=True)


    tokenized_datasets = {'train': train_data, 'eval': eval_data}

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    batch_size = 8
    model_name = args.model_name.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-{args.kode_simpan}",
        overwrite_output_dir = True,
        eval_strategy= "epoch",
        save_strategy= "epoch", 
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
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
        eval_dataset=tokenized_datasets["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(tokenizer)
    )
                            
    trainer.train()
    trainer.push_to_hub()

    return

if __name__ == "__main__":
    main()

