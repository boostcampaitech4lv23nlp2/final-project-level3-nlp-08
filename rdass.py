from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def rdass(generation, answer, context):
    g_emb = model.encode(generation)
    a_emb = model.encode(answer)
    d_emb = model.encode(context)

    ga_cos = util.cos_sim(g_emb, a_emb)
    gd_cos = util.cos_sim(g_emb, d_emb)

    return (ga_cos + gd_cos) / 2