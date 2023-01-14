from konlpy.tag import Mecab
from krwordrank.word import KRWordRank

import re

m = Mecab()

def preprocessing(context):
    context = re.sub('\n','',context)
    context = re.sub('\u200b','',context)
    context = re.sub('\xa0','',context)
    context = context.replace('</s>', '.')
    context = re.sub('([a-zA-Z])','',context)
    context = re.sub('[ㄱ-ㅎㅏ-ㅣ]+','',context)
    context = re.sub('[\\‘|\(\)\[\]\<\>`\'…》]','',context)

    return context

def makeNounList(context):
    pos_list = m.pos(context)

    noun_list = []
    for pos in pos_list:
        if pos[1] == 'NNG':
            noun_list.append(pos[0])
    
    return noun_list
context = '<s>진짜 심각하다 아니 바람은 오지게부는데 더워</s> <s>그러게 에어컨 틀어져잇어서 안도의 한숨 쉼 휴 오늘 셔츠 괜히입엇다</s> <s>나 자켓입엇는데 개더워서 과장님한테 선풍기 빌려주세요 햇는데 과장님 쓰고 계셧어... 근데 내가 지금 쓰구잇어...</s> <s>당신의 인성.. 어디까지 가야 더 밑바닥이 나올까요</s> <s>ㅋㅋㅋㅋㅋㅋㅋㅋㅋ과장님이 빌랴주신거야..</s>'

context = preprocessing(context)
nounList = makeNounList(context)
context_list = context.split('.')

min_count = 3
max_length = 10
wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length)

beta = 0.85
max_iter = 10
keywords, rank, graph = wordrank_extractor.extract(context_list, beta, max_iter)
print(keywords)
answer = []
for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:30]:
        print('%8s:\t%.4f' % (word, r))
        if word in nounList:
            answer.append(word)

print(answer)


