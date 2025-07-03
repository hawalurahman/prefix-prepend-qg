#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

with open('squad-id-eval.json', 'r') as f:
    data = json.load(f)


# In[35]:


get_ipython().system('nvidia-smi')


# In[3]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# # SQUAD Paragraph

# In[142]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained('hawalurahman/mt5-base-qaqg-finetuned-SQuAD-id')
model = AutoModelForSeq2SeqLM.from_pretrained('hawalurahman/mt5-base-qaqg-finetuned-SQuAD-id').to('cuda')


# In[145]:


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


data_qa = data_qa[:500]
data_qg = data_qg[:500]


# In[146]:


len(data_qg)


# In[148]:


from tqdm import tqdm


predictions_qa = []
references_qa = []

for i, each in enumerate(tqdm(data_qa)):
    text = each['context']
    inputs = tokenizer(text, return_tensors="pt").input_ids.to('cuda')

    outputs = model.generate(inputs, max_new_tokens = 200)
    outputs_txt = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pred_ans = [item.strip() for item in outputs_txt.split("[SEP]")]
    pred_ans = ' '.join(pred_ans) # semua hasil per jawaban itu digabungkan
    ref_ans = [item.strip() for item in each['target'].split("[SEP]")] # kunci jawaban referensi juga disatukan supaya mudah
    ref_ans = ' '.join(ref_ans)

    predictions_qa.append(pred_ans)
    references_qa.append(ref_ans)

#     print(pred_ans)
#     print(ref_ans)

#     if i == 1:
#         break

predictions_qg = []
references_qg = []

for i, each in enumerate(tqdm(data_qg)):
    text = each['context']
    inputs = tokenizer(text, return_tensors="pt").input_ids.to('cuda')

    outputs = model.generate(inputs, max_new_tokens = 200)
    outputs_txt = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pred_ans = [item.strip() for item in outputs_txt.split("[SEP]")]
    pred_ans = ' '.join(pred_ans) # semua hasil per jawaban itu digabungkan
    ref_ans = each['target'] # kunci jawaban referensi juga disatukan supaya mudah

    predictions_qg.append(pred_ans)
    references_qg.append(ref_ans)

#     print(pred_ans)
#     print(ref_ans)

#     if i == 1:
#         break



# In[149]:


import evaluate

# # Pad predictions with None or an empty string
# while len(predictions) < len(references):
#     predictions.append('')  # or predictions.append('')

rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")

bleu_results_qa = bleu.compute(predictions=predictions_qa, references=references_qa, tokenizer=word_tokenize)
bleu_results_qg = bleu.compute(predictions=predictions_qg, references=references_qg, tokenizer=word_tokenize)

rouge_results_qa = rouge.compute(predictions=predictions_qa, references=references_qa, tokenizer=word_tokenize)
rouge_results_qg = rouge.compute(predictions=predictions_qg, references=references_qg, tokenizer=word_tokenize)


# In[150]:


print('squd paragraph')
print('-'*100)
print('qa', bleu_results_qa, rouge_results_qa)
print('-'*100)
print('qg', bleu_results_qg, rouge_results_qg)


# # SQUAD Sentence

# In[151]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained('hawalurahman/mt5-base-qaqg-finetuned-SQuAD-id-sentence')
model = AutoModelForSeq2SeqLM.from_pretrained('hawalurahman/mt5-base-qaqg-finetuned-SQuAD-id-sentence').to('cuda')

# tokenizer3 = AutoTokenizer.from_pretrained('hawalurahman/mt5-base-qaqg-finetuned-SQuAD-id-ir')
# model3 = AutoModelForSeq2SeqLM.from_pretrained('hawalurahman/mt5-base-qaqg-finetuned-SQuAD-id-ir').to('cuda')


# In[21]:


data


# In[154]:


from nltk import sent_tokenize

data_qa = []
data_qg = []

# membuat data untuk answer extraction
# context = prefix+context --> answers = item [SEP] item [SEP]
for i, each in enumerate(data):
    answers = ' [SEP] '.join(each['answers'])
    context = 'extract answer: '+each['context']

    data_qa.append({'context': context, 'target': answers})

# membuat data untuk question generation, dengan sentence yang lebih sedikit
# context = prefix+context [SEP] answer --> question = question
for i, each in enumerate(data):
    for j, answer in enumerate(each['answers']):
        answer_txt = answer.strip()
        context_sents = sent_tokenize(each['context'])

        for k, sent in enumerate(context_sents):
            if answer_txt in sent:
#                 print(answer_txt, sent)

                context = f"generate question: {sent} [SEP] {answer}"
                question = each['questions'][j]
#                 print(context)

                data_qg.append({'context': context, 'target': question})
                pass

data_qa = data_qa[:500]
data_qg = data_qg[:500]


# In[29]:


len(data_qg)


# In[30]:


data_qg


# In[155]:


from tqdm import tqdm


predictions_qa = []
references_qa = []

for i, each in enumerate(tqdm(data_qa)):
    text = each['context']
    inputs = tokenizer(text, return_tensors="pt").input_ids.to('cuda')

    outputs = model.generate(inputs, max_new_tokens = 200)
    outputs_txt = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pred_ans = [item.strip() for item in outputs_txt.split("[SEP]")]
    pred_ans = ' '.join(pred_ans) # semua hasil per jawaban itu digabungkan
    ref_ans = [item.strip() for item in each['target'].split("[SEP]")] # kunci jawaban referensi juga disatukan supaya mudah
    ref_ans = ' '.join(ref_ans)

    predictions_qa.append(pred_ans)
    references_qa.append(ref_ans)

#     print(pred_ans)
#     print(ref_ans)

#     if i == 1:
#         break

predictions_qg = []
references_qg = []

for i, each in enumerate(tqdm(data_qg)):
    text = each['context']
    inputs = tokenizer(text, return_tensors="pt").input_ids.to('cuda')

    outputs = model.generate(inputs, max_new_tokens = 200)
    outputs_txt = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pred_ans = [item.strip() for item in outputs_txt.split("[SEP]")]
    pred_ans = ' '.join(pred_ans) # semua hasil per jawaban itu digabungkan
    ref_ans = each['target'] # kunci jawaban referensi juga disatukan supaya mudah

    predictions_qg.append(pred_ans)
    references_qg.append(ref_ans)

#     print(pred_ans)
#     print(ref_ans)

#     if i == 1:
#         break



# In[156]:


import evaluate

# # Pad predictions with None or an empty string
# while len(predictions) < len(references):
#     predictions.append('')  # or predictions.append('')

rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")

bleu_results_qa = bleu.compute(predictions=predictions_qa, references=references_qa, tokenizer=word_tokenize)
bleu_results_qg = bleu.compute(predictions=predictions_qg, references=references_qg, tokenizer=word_tokenize)

rouge_results_qa = rouge.compute(predictions=predictions_qa, references=references_qa, tokenizer=word_tokenize)
rouge_results_qg = rouge.compute(predictions=predictions_qg, references=references_qg, tokenizer=word_tokenize)


# In[158]:


print('squad sentence')
print('-'*100)
print('qa', bleu_results_qa, rouge_results_qa)
print('-'*100)
print('qg', bleu_results_qg, rouge_results_qg)


# # Squad IR

# In[159]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_checkpoint = "hawalurahman/mt5-base-qaqg-finetuned-SQuAD-id-ir"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to('cuda')


# In[6]:


data


# In[160]:


from tqdm import tqdm
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

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
# pakai bm25 untuk retrive sentence.
def rank_sentences_bm25(query, doc):
    # Tokenize the query and document sentences
    tokenized_query = word_tokenize(query.lower())
    tokenized_doc = [word_tokenize(sentence.lower()) for sentence in doc]

    # Initialize BM25 with the tokenized document
    bm25 = BM25Okapi(tokenized_doc)

    # Get BM25 scores for the query against the document sentences
    scores = bm25.get_scores(tokenized_query)

    # Rank sentences by score
    ranked_sentences = sorted(zip(scores, doc), reverse=True)

    return ranked_sentences


for i, each in enumerate(tqdm(data)):
    for j, answer in enumerate(each['answers']):
        sentences = sent_tokenize(each['context'])
        ranked_sentences = rank_sentences_bm25(answer, sentences)
        doc_only = [sentence for _, sentence in ranked_sentences]
        new_context = " ".join(doc_only[:2])

#         print(answer, doc_only[:2])

        context = f"generate question: {new_context} [SEP] {answer}"
        question = each['questions'][j]

        data_qg.append({'context': context, 'target': question})



data_qa = data_qa[:500]
data_qg = data_qg[:500]        


# In[161]:


len(data_qg)


# In[163]:


from tqdm import tqdm


predictions_qa = []
references_qa = []

for i, each in enumerate(tqdm(data_qa)):
    text = each['context']
    inputs = tokenizer(text, return_tensors="pt").input_ids.to('cuda')

    outputs = model.generate(inputs, max_new_tokens = 200)
    outputs_txt = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pred_ans = [item.strip() for item in outputs_txt.split("[SEP]")]
    pred_ans = ' '.join(pred_ans) # semua hasil per jawaban itu digabungkan
    ref_ans = [item.strip() for item in each['target'].split("[SEP]")] # kunci jawaban referensi juga disatukan supaya mudah
    ref_ans = ' '.join(ref_ans)

#     print(pred_ans)
# #     print(each['context'])
#     print(ref_ans)

    predictions_qa.append(pred_ans)
    references_qa.append(ref_ans)

#     print(pred_ans)
#     print(ref_ans)

#     if i == 1:
#         break




predictions_qg = []
references_qg = []

for i, each in enumerate(tqdm(data_qg)):
    text = each['context']
    inputs = tokenizer(text, return_tensors="pt").input_ids.to('cuda')

    outputs = model.generate(inputs, max_new_tokens = 200)
    outputs_txt = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pred_ans = [item.strip() for item in outputs_txt.split("[SEP]")]
    pred_ans = ' '.join(pred_ans) # semua hasil per jawaban itu digabungkan
    ref_ans = each['target'] # kunci jawaban referensi juga disatukan supaya mudah

    predictions_qg.append(pred_ans)
    references_qg.append(ref_ans)

#     print(pred_ans)
#     print(ref_ans)

#     if i == 1:
#         break



# In[164]:


import evaluate

# # Pad predictions with None or an empty string
# while len(predictions) < len(references):
#     predictions.append('')  # or predictions.append('')

rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")

bleu_results_qa = bleu.compute(predictions=predictions_qa, references=references_qa, tokenizer=word_tokenize)
bleu_results_qg = bleu.compute(predictions=predictions_qg, references=references_qg, tokenizer=word_tokenize)

rouge_results_qa = rouge.compute(predictions=predictions_qa, references=references_qa, tokenizer=word_tokenize)
rouge_results_qg = rouge.compute(predictions=predictions_qg, references=references_qg, tokenizer=word_tokenize)


# In[165]:


print('squad pakai IR')
print('-'*100)
print('qa', bleu_results_qa, rouge_results_qa)
print('-'*100)
print('qg', bleu_results_qg, rouge_results_qg)

