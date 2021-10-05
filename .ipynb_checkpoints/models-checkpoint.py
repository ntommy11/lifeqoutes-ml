import torch

from transformers import AutoTokenizer, AutoConfig, AutoModel, Trainer, TrainingArguments

'''
def RoBERTa(num_labels=30):
    MODEL_NAME = 'klue/roberta-large'
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = num_labels
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    special_tokens_dict = {
        'additional_special_tokens':[
            '[SUB]',
            '[/SUB]',
            '[OBJ]',
            '[/OBJ]'
        ]
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print("num_added_tokens:",num_added_tokens)
    
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer
'''

def get_tokenizer(MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens_dict = {
        'additional_special_tokens':[
            '[SUB:ORG]',
            '[SUB:PER]',
            '[/SUB]',
            '[OBJ:DAT]',
            '[OBJ:LOC]',
            '[OBJ:NOH]',
            '[OBJ:ORG]',
            '[OBJ:PER]',
            '[OBJ:POH]',
            '[/OBJ]'
        ]
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print("num_added_tokens:",num_added_tokens)
    return tokenizer

def load_RoBERTa():
    model = RoBERTa()
    return model

class RoBERTa(torch.nn.Module):
    def __init__(self,num_labels):
        super().__init__()
        self.MODEL_NAME = 'klue/roberta-large'
        self.bert_model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.hidden_size = 1024
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        
        special_tokens_dict = {
            'additional_special_tokens':[
                '[SUB:ORG]',
                '[SUB:PER]',
                '[/SUB]',
                '[OBJ:DAT]',
                '[OBJ:LOC]',
                '[OBJ:NOH]',
                '[OBJ:ORG]',
                '[OBJ:PER]',
                '[OBJ:POH]',
                '[/OBJ]'
            ]
        }

        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        print("num_added_tokens:",num_added_tokens)

        self.bert_model.resize_token_embeddings(len(self.tokenizer))
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(3*self.hidden_size,self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.1, inplace=False),
            torch.nn.Linear(self.hidden_size,self.num_labels)
        )
        
    def forward(self,item):
        input_ids = item['input_ids']
        token_type_ids = item['token_type_ids']
        attention_mask = item['attention_mask']
        sub_token_index = item['sub_token_index']
        obj_token_index = item['obj_token_index']
        out = self.bert_model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        h = out.last_hidden_state
        #print(h.shape)
        batch_size = h.shape[0]
        
        stack = []
        
        for i in range(batch_size):
            stack.append(torch.cat([h[i][0],h[i][sub_token_index[i]],h[i][obj_token_index[i]]]))
        
        stack = torch.stack(stack)
                                
        #print("stack:",stack.shape)
        out = self.classifier(stack)
        return out
        