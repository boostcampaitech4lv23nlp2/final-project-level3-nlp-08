import torch
from torch import nn
from transformers import (PreTrainedTokenizerFast, 
                          AutoConfig, 
                          BartForConditionalGeneration)

    
class BaseModel(nn.Module):
    def __init__(self, MODEL_NAME:str):
        super().__init__()
        self.MODEL_NAME = MODEL_NAME
        # plm 모델 설정
        self.plm = BartForConditionalGeneration.from_pretrained(self.MODEL_NAME)
        self.plm.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.MODEL_NAME)
        self.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, inputs):

        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        if 'inputs_embeds' in inputs:
            inputs['input_ids'] = None
        else:
            inputs['inputs_embeds'] = None
        
        return self.plm(input_ids=inputs['input_ids'],
                        inputs_embeds=inputs['inputs_embeds'],
                        attention_mask=attention_mask,
                        decoder_input_ids=inputs['decoder_input_ids'],
                        decoder_attention_mask=decoder_attention_mask,
                        labels=inputs['labels'], return_dict=True)
    
    
class SubjectModel(nn.Module):
    def __init__(self, BART_MODEL_NAME:str, cnn_channel=128, token_length=512):
        super().__init__()
        self.BART_MODEL_NAME = BART_MODEL_NAME
        # plm 모델 설정
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.BART_MODEL_NAME)
        self.config = AutoConfig.from_pretrained(self.BART_MODEL_NAME)
        self.bart_encoder = BartForConditionalGeneration.from_pretrained(self.BART_MODEL_NAME).model.encoder
        # freeze
        for param in self.bart_encoder.parameters():
                param.requires_grad = False

        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.token_length = token_length

        self.pad_token_id = self.tokenizer.pad_token_id
        
        self.cnn_layers = nn.ModuleList([nn.Conv1d(in_channels=self.config.d_model, 
                                        out_channels=cnn_channel, kernel_size=i) for i in range(3, 11, 2)])
        self.pooling_layers = nn.ModuleList([nn.MaxPool1d(self.token_length - i + 1) for i in range(3, 11, 2)])
        self.linear = nn.Linear(cnn_channel*4, 20)
            
    def forward(self, inputs):
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        encoder_out = self.bart_encoder(input_ids=inputs['input_ids'],
                        attention_mask=attention_mask,
                        return_dict=True)['last_hidden_state']
        encoder_out = encoder_out.transpose(1, 2)
        tmp = []
        for i in range(len(self.cnn_layers)):
            t = torch.relu(self.cnn_layers[i](encoder_out))
            t = self.pooling_layers[i](t)
            tmp.append(t)

        y = torch.cat(tmp, axis=1).squeeze() 

        logits = self.linear(y)
        return {'logits': logits, 'embedding': y}