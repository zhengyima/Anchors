import torch
import torch.utils.checkpoint
from torch import nn
from transformers import BertForPreTraining

from transformers.models.bert.modeling_bert import BertForPreTrainingOutput
from torch.nn import MarginRankingLoss

class PairWiseBertForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None

        if labels is not None and next_sentence_label is not None:
            seq_relationship_logits = seq_relationship_score # bs, 2
            seq_relationship_scores = seq_relationship_logits[:,1] # bs, 1

            batch_size = seq_relationship_logits.size(0)
            softmax = nn.Softmax(dim=1)
            marginloss = MarginRankingLoss(margin=1.0, reduction='mean')
            logits = seq_relationship_scores.reshape(batch_size//2, 2) # bs/2, 2
            logits = softmax(logits)
            pos_logits = logits[:, 0] #bs/2, 1
            neg_logits = logits[:, 1] # bs/2, 1
            pairwise_label = torch.ones_like(pos_logits) # bs/2, 1
            pairwise_loss = marginloss(pos_logits, neg_logits, pairwise_label)

            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            total_loss = pairwise_loss + masked_lm_loss
            
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output        

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        

