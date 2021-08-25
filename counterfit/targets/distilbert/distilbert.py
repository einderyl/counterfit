import torch
import pandas as pd
from counterfit.core.targets import TextTarget
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch.nn.functional as F


class Distilbert(TextTarget):
    data_path = 'drive/MyDrive/textstosave.txt'
    model_name = "db"
    model_data_type = "text"
    model_endpoint = "drive/MyDrive/distilbert"
    model_input_shape = (1,)
    model_output_classes = [0,1]
    X = []

    def __init__(self):
        self.X = self._get_data(self.data_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_endpoint, num_labels=2) 
        
    @staticmethod
    def _get_data(file_loc):
        with open(file_loc, 'r') as f:
            text = f.read()
            
        texts = text.split('\n')
        return texts

    def __call__(self, x):
        print(x.shape)
        print(x[0])
        new_x = x.squeeze().tolist() if len(x.shape) == (2) else x
        encoded_dict = self.tokenizer.batch_encode_plus(
                        new_x,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        truncation = True,
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )
        with torch.no_grad():
            result = self.model(encoded_dict['input_ids'], attention_mask=encoded_dict['attention_mask'], return_dict=True)
        logits = result.logits
        softmax_output = F.softmax(logits, dim=1).numpy()
        return softmax_output
