from keybert import KeyBERT

context = '제주도에 고기국수가 맛있는데 있는지 찾아봐야 할 것 같다.제주도 흑돼지는 숙성도 좋고 말고기는 마우돈이 괜찮다고 해서 먹어보고 싶다고 이야기한다.'

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(context)
print(keywords)