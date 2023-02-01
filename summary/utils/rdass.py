from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer('jhgan/ko-sroberta-multitask')
device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
model.to(device)

def sim(generation, answer, context):
    g_emb = model.encode(generation)
    a_emb = model.encode(answer)

    ga_cos = util.cos_sim(g_emb, a_emb)

    return ga_cos

    
def rdass(generation, answer, context):
    g_emb = model.encode(generation)
    a_emb = model.encode(answer)
    d_emb = model.encode(context)

    ga_cos = util.cos_sim(g_emb, a_emb)
    gd_cos = util.cos_sim(g_emb, d_emb)

    return (ga_cos + gd_cos) / 2