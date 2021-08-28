from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.init as init

class BertForCLSWeight(nn.Module):
    def __init__(self, bert_model):
        super(BertForCLSWeight, self).__init__()
        self.bert_model = bert_model
    
    def forward(self, batch_data):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """
        # attentions: bs, num_heads, sl, sl
        bert_output = self.bert_model(**batch_data)
        attentions = bert_output[2][-1] # attentions, last layer
        # print(attentions.size())
        # print(bert_output[2][-1].size())
        attentions = torch.sum(attentions, dim=1) # bs, sl, sl 
        attentions = attentions[:,0,:] # bs, sl
        return attentions

class BertForAnchorWeight(nn.Module):
    def __init__(self, bert_model):
        super(BertForAnchorWeight, self).__init__()
        self.bert_model = bert_model
    
    def forward(self, batch_data):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """
        # attentions: bs, num_heads, sl, sl
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        token_type_ids = batch_data['token_type_ids']
        aidx = batch_data['aidx']
        bert_input = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

        bert_output = self.bert_model(**bert_input)
        attentions = bert_output[2][-1] # attentions, last layer
        # print(attentions.size())
        # print(bert_output[2][-1].size())
        attentions = torch.sum(attentions, dim=1) # bs, sl, sl 
        attentions = attentions[:,:,:] # bs, sl
        return attentions
        
