import json
import pandas as pd
from datasets import load_from_disk,DatasetDict, Dataset
import re
import random
from collections import OrderedDict
from pymongo import MongoClient
random.seed(42)

def make_sentence(text):
    a=text.split('\n')
    tmp=[]
    for c in a:
        tmp += re.split(r'([.!?])',c)
    tmp2 = []
    special=['.','!','?']
    for i in range(len(tmp)-1):
        if tmp[i] in special:
            continue
        elif tmp[i] == '':
            continue
        else:
            if tmp[i+1] in special:
                tmp2.append(tmp[i]+tmp[i+1])
                i += 1
            else:
                tmp2.append(tmp[i])
    return tmp2

client = MongoClient("mongodb+srv://nlp-08:finalproject@cluster0.rhr2bl2.mongodb.net/?retryWrites=true&w=majority")
db=client['test_database']
blogs=db.blogs
y = blogs.find()

# ICT make for train_dataset
i=0
error_list=[]
titles = []
contexts = []
questions = []
answers = []
ids = []
for val in y:
    try:
        sentences=make_sentence(val['content'])
        target = random.randint(0,len(sentences)-1)
        target_senteces = ' '.join(sentences[target:target+3])
        target_context = ' '.join(sentences[:target]+sentences[target+3:])
        titles.append(val['title'])
        contexts.append(target_context)
        questions.append(target_senteces)
        answers.append(val['content'])
        ids.append(f'{i}')
        i += 1
    except:
        error_list.append(val)

blog_df = pd.DataFrame({'title':titles,'context': contexts, 'question': questions,'ground_truth':answers, 'id': ids})
blog_df = blog_df.drop_duplicates(['context','question'],keep='first',ignore_index=True)

train_dataset = Dataset.from_pandas(blog_df[:78711])
valid_dataset = Dataset.from_pandas(blog_df[78711:])
datasets = DatasetDict({'train': train_dataset, 'validation': valid_dataset})
datasets.save_to_disk('../../../json_data/new_blogs_ict_dataset')