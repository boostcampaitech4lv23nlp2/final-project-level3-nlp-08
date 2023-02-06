from pymongo import MongoClient
client = MongoClient("mongodb+srv://nlp-08:finalproject@cluster0.rhr2bl2.mongodb.net/?retryWrites=true&w=majority")
db=client['test_database']
print(client.list_database_names())
blogs=db.blogs

import numpy as np
from model import *
from tokenizer import *
from transformers import AutoTokenizer

MODEL_NAME="LeeHJ/colbert_blog"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ColbertModel.from_pretrained(MODEL_NAME).to('cuda')

p_embs = []
for v in blogs.find():
    with torch.no_grad():
        model.eval()
        p = v['content']
        p = tokenize_colbert(p, tokenizer, corpus="doc").to("cuda")
        p_emb = model.doc(**p).to("cpu").numpy()
        p_embs.append({str(v['_id']): p_emb})

print("emb len : ",len(p_embs))
np.save('embs.npy',p_embs)

t=np.load('embs.npy',allow_pickle=True)
import os
f = os.path.getsize('embs.npy')

def convert_size(size_bytes):
    import math
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

print('File_size',convert_size(f),'bytes')