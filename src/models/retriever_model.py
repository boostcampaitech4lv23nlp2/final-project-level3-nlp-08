from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
sys.path.append('/opt/ml/final-project-level3-nlp-08/src/elastic')
from elastic import ElasticObject
import torch
import time
import json
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from model import ColbertModel
from tqdm import tqdm, trange
import re
import numpy as np
import random

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("LeeHJ/colbert_blog")

elastic_connector = ElasticObject("localhost:9200")

def tokenize_colbert(dataset, tokenizer ,corpus):

    # for inference
    if corpus == "query":
        preprocessed_data = "[Q] " + preprocess(dataset)

        tokenized_query = tokenizer(
            preprocessed_data, return_tensors="pt", padding=True, truncation=True, max_length=128,
        )
        return tokenized_query

    elif corpus == "doc":
        preprocessed_data = "[D] " + preprocess(dataset)
        tokenized_context = tokenizer(
            preprocessed_data,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return tokenized_context

    elif corpus == "bm25_hard":
        preprocessed_context = []
        for context in dataset:
            preprocessed_context.append("[D] " + preprocess(context))
        tokenized_context = tokenizer(
            preprocessed_context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return tokenized_context
    # for train
    else:
        preprocessed_query = []
        preprocessed_context = []
        for query, context in zip(dataset["query"], dataset["context"]):
            preprocessed_context.append("[D] " + preprocess(context))
            preprocessed_query.append("[Q] " + preprocess(query))
        tokenized_query = tokenizer(
            preprocessed_query, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        tokenized_context = tokenizer(
            preprocessed_context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return tokenized_context, tokenized_query

def preprocess(text):
    text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”"":&<>·\-\'+\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # 두 개 이상의 연속된 공백을 하나로 치환
    return text  

def tokenize(query, tokenizer):
    # print(query)
    preprocessed_data="[Q] " + preprocess(query)
    tokenized_query = tokenizer(
        preprocessed_data, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    return tokenized_query

def get_score(Q, D, N=None, eval=False):
        # hard negative N은 train에만 쓰임.
        if eval:
            final_score = torch.tensor([])
            for D_batch in tqdm(D):
                D_batch = np.array(D_batch)
                D_batch = torch.Tensor(D_batch).squeeze()
                print("Docu_dim_size!! : ",D_batch.shape)
                p_seqeunce_output = D_batch.transpose(
                    1, 2
                )  # (batch_size,hidden_size,p_sequence_length)-> 200 128 512
                print("Query_dim_size!! : ",Q.size()) 
                q_sequence_output = Q.view(
                    1, 1, -1, 128
                )  # (1, 1, q_sequence_length, hidden_size)
                dot_prod = torch.matmul(
                    q_sequence_output, p_seqeunce_output
                )  # (1,batch_size, q_sequence_length, p_seqence_length)
                max_dot_prod_score = torch.max(dot_prod, dim=3)[
                    0
                ]  # (batch_size,batch_size,q_sequnce_length)
                score = torch.sum(max_dot_prod_score, dim=2)  # (batch_size,batch_size)
                final_score = torch.cat([final_score, score], dim=1)
            print("final_score!! :",final_score.size())
            return final_score

def load_model():
    col_model = ColbertModel.from_pretrained("LeeHJ/colbert_blog").to('cuda')
    col_model.to(device)
    col_model.eval()
    
    return col_model

model = load_model()

def retriever(query):
    _,outputs = elastic_connector.search(index_name="blogs", question=query, topk=100)
    print("outputs!!!",outputs)
    print("wwwwwwwwwwwww!!!",type(outputs),outputs)
    return outputs



class ServerHandler(BaseHTTPRequestHandler):
    summary_messages = ["지금까지 대화한 내용을 요약해 봤어!", "멍멍멍멍멍멍(대충 요약했다는 뜻)", "너네 이런 대화했지? 맞췄지? 잘했지?"]
    recommend_messages = ["내가 추천해 주는 글이 도움이 될거야!", "내가 열심히 찾아봤다 멍멍!", "이게 좋겠다!", "멍머멍(대충 문서를 가져왔다는 뜻)"]

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
    
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        message = json.loads(self.rfile.read(length))
        
        input_text = message['text']
        start_time = time.time()
        infer_text = retriever(input_text) #100개 뽑기
        infer_time = time.time() - start_time
        
        print("infer_text[source!!!",infer_text['source'])
        contexts = [output['_source']['content'] for output in infer_text['source']]
        urls = [output['_source']['url'] for output in infer_text['source']]
        titles = [output['_source']['title'] for output in infer_text['source']]
        print("contexts!!!",len(contexts))

        batched_p_embs = []

        with torch.no_grad():
            print("Start passage embedding......")
            p_embs = []
            for step, p in enumerate(tqdm(contexts)):
                p = tokenize_colbert(p, tokenizer, corpus="doc").to("cuda")
                p_emb = model.doc(**p).to("cpu").numpy()
                p_embs.append(p_emb)
                if (step + 1) % 200 == 0:
                    batched_p_embs.append(p_embs)
                    p_embs = []
            if p_embs:
                batched_p_embs.append(p_embs)

        with torch.no_grad():
            model.eval()
            q_seqs_val = tokenize(input_text, tokenizer).to("cuda")
            q_emb = model.query(**q_seqs_val).to("cpu")
        
        dot_prod_scores = get_score(q_emb, batched_p_embs, eval=True)
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
                    
        sources = []
        for i in range(3):
            source_data = infer_text['source'][rank[i]]
            sources.append(source_data)
        
        random_summary_idx = random.randint(0, len(self.summary_messages)-1)
        random_recommend_idx = random.randint(0, len(self.recommend_messages)-1)
        
        response = {
            "location": "recommend",
            "summary_message": self.summary_messages[random_summary_idx],
            "summary": input_text,
            "recommend_message": self.recommend_messages[random_recommend_idx],
            "source": sources
        }
        message['answer'] = response     
        #message['answer'] = infer_text
        message['took'] = infer_time
        #print("response!",message)
        self._set_headers()
        self.wfile.write(json.dumps(message).encode())
        
    
def run(server_class=HTTPServer, handler_class=ServerHandler, port=8503):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting inference module on port %d...' % port)
    httpd.serve_forever()

if __name__ == "__main__":
    run()
    # print(generate('후쿠오카 어쩌구 저쩌구 이거 요약해주라고 좋냐 안좋냐후쿠오카 일본일본 어떄?일본 별루?일본짱엥?렉걸린다쉣일본 별로야?????????????????????일본 맛집 어떤데???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????'))