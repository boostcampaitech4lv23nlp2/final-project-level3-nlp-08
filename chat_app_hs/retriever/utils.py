from .model import *
from .tokenizer import *

def preprocess(text):
    text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”"":&<>·\-\'+\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # 두 개 이상의 연속된 공백을 하나로 치환
    return text  

def tokenize(query, tokenizer):
    print(query)
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