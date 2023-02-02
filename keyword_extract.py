from keybert import KeyBERT

context = '<s>진짜 심각하다 아니 바람은 오지게부는데 더워</s> <s>그러게 에어컨 틀어져잇어서 안도의 한숨 쉼 휴 오늘 셔츠 괜히입엇다</s> <s>나 자켓입엇는데 개더워서 과장님한테 선풍기 빌려주세요 햇는데 과장님 쓰고 계셧어... 근데 내가 지금 쓰구잇어...</s> <s>당신의 인성.. 어디까지 가야 더 밑바닥이 나올까요</s> <s>ㅋㅋㅋㅋㅋㅋㅋㅋㅋ과장님이 빌랴주신거야..</s>'

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(context)
print(keywords)