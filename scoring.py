import pandas as pd 
from rouge import Rouge

import torch
from transformers.models.bart import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./models/kobart_summary_concatenate_new')
    return model

model = load_model()
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')

test_data = pd.read_csv('./data_csv_new/Test/all_05.csv')
contexts, summarys = [], []

rouge = Rouge()

for context, summary in zip(test_data['context'], test_data['summary']):
    contexts.append(context)
    summarys.append(summary)

num = len(contexts)

rouge_1, rouge_2, rouge_l = 0, 0, 0
for c, s in tqdm(zip(contexts, summarys)):
    input_ids = tokenizer.encode(c)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    score = rouge.get_scores(output, s, avg=True)

    rouge_1 += score['rouge-1']['f']
    rouge_2 += score['rouge-2']['f']
    rouge_l += score['rouge-l']['f']

rouge_1 /= num
rouge_2 /= num
rouge_l /= num

print(f'ROUGE-1: {rouge_1} | ROUGE-2: {rouge_2} | ROUGE-L: {rouge_l}')
