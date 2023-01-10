import torch
from transformers.models.bart import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    return model

model = load_model()
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')

text = '음, 우선 정보가 부족한데요, 말씀해주신 사항만 바탕으로해서 고민한다면 아래와 같은 지점들을 중요시 볼 것 같습니다.문제정의 -> 이 task를 왜 하게됐는가? 왜 필요한가? 수요가 있는가?. 중요한것은 method와 목표가 혼동되어서는 안된다고 생각합니다. 말씀해주신대로, ‘회의 요약‘은 method거든요. 주제를 정하고, 문제점을 분석하고, 그 문제점을 극복하는 방법이 method가 될 것입니다. 문제 정의가 명확하고, 타당하다면 거기서 사업성을 찾는 것은 아주 쉬울 것입니다.‘요약‘에서 발생할 수 있는 문제점들을 어떻게 극복했는가? -> ‘요약‘의 특성상, 사람마다 중요시하는 요소가 다를 것입니다. 예를 들어 저는 일정 산출을 중요시하게 생각하는 반면, 누군가는 일정보다는 큰 주제와 흐름의 타당성을 중요시생각할 수도 있겠죠.서비스 수준까지 성능을 끌어올리기 위해 어떤 노력을 했는가? -> 가설 설정 -> 실험 -> 문제 직면 -> 극복 방안 가설 설정 -> 실험 -> 문제직면 … (반복) -> 최종'
text = text.replace('\n', '')

input_ids = tokenizer.encode(text)
input_ids = torch.tensor(input_ids)
input_ids = input_ids.unsqueeze(0)
output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
output = tokenizer.decode(output[0], skip_special_tokens=True)

print(output)