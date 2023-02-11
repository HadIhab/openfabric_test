import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

import pandas as pd
import numpy as np
import torch

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass
############################################################
# function used for download the model & the tokenizer
# then load them.
############################################################
def model_load():
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    model_weights_dir = './model_weights/'

    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)

    model_weights_files = ['config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer_config.json',
                           'vocab.txt']
    weights_files_present = all(os.path.exists(os.path.join(model_weights_dir, f)) for f in model_weights_files)

    if weights_files_present:
        print('[INFO]: Loading model & tokenizer from local machine please wait..')
        model = BertForQuestionAnswering.from_pretrained(model_weights_dir)
        tokenizer = BertTokenizer.from_pretrained(model_weights_dir)
        print('[INFO]: Loading model and tokenizer finished')
    else:
        print('[INFO]: Downloading model & tokenizer please wait until download complete..')
        model = BertForQuestionAnswering.from_pretrained(model_name)
        model.save_pretrained(model_weights_dir)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_weights_dir)
        print('[DOWNLOAD]: Downloading model & tokenizer finished')
    return model, tokenizer

def answer_question(question, text):
    print('[INFO]:  Your request arrived ..')
    model, tokenizer = model_load()

    input_ids = tokenizer.encode(question, text)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_idx = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_idx + 1
    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = " ".join(tokens[answer_start:answer_end + 1])
    else:
        return "Unfortunately I can not find answer, please re-formulate your question."
    return answer

############################################################
# Callback function called on each execution pass
############################################################
def execute(request, ray: OpenfabricExecutionRay):
        output = []
        for text in request["question"]:
            response = answer_question(text,request["text"])
            output.append(response)
        return output[0]
