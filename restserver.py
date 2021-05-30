#!/usr/bin/env python

# WS server example that synchronizes state across clients

# https://towardsdatascience.com/how-to-use-bert-from-the-hugging-face-transformer-library-d373a22b0209

from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from torch.nn import functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello_world():
    # return 'oh no'
    data = request.get_json(force=True)
    message = data['message']
    sentences = data['sentences']

    response = []

    for i, sentence in enumerate(sentences):
        encoding = tokenizer.encode_plus(sentence, message, return_tensors='pt')
        outputs = model(**encoding)[0]
        a, b = toVec(outputs)

        softmax = F.softmax(outputs, dim = 1)

        norm_a, norm_b = toVec(softmax)

        response.append({
            'index' : i,
            'sentence' : sentence,
            'score' : float(a),
            'rand': float(b),
            'norm_score' : float(norm_a),
            'norm_rand' : float(norm_b),
            # 'incoherence' : double(incoherence)
        })

    return jsonify(response)

def toVec(output):
    o = str(output).split('[[', 1)[1].split(']]', 1)[0]
    a, b = [x.strip() for x in o.split(',')]
    return [a, b]
