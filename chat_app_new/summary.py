import torch
from transformers.models.bart import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast

def load_model():
    model = BartForConditionalGeneration.from_pretrained('papari1123/summary_bart_dual_R3F_aihub')
    return model

model = load_model()
tokenizer = PreTrainedTokenizerFast.from_pretrained('papari1123/summary_bart_dual_R3F_aihub')

def summarize(text):
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    return output