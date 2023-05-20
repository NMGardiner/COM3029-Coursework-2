# Model
from transformers import AutoTokenizer, BertForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("model")
model = BertForSequenceClassification.from_pretrained("model")

labels = ["admiration", "gratitude", "amusement", "love", "optimism", "joy", "annoyance", "disapproval", "anger", "sadness", "remorse", "curiosity", "surprise", "neutral"]

def predict_label(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model(**inputs)
    
    label_index = torch.argmax(outputs.logits, dim=1).item()

    return labels[label_index]
    
# Logging
import json
import logging
from logging.config import dictConfig
from datetime import datetime

dictConfig({
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(message)s",
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "predictions.txt",
            "formatter": "default",
        },
    },
    "root": {"level": "INFO", "handlers": ["file"]},
})

# Prevent regular app messages getting logged.
logging.getLogger("werkzeug").disabled = True

# Use Python's structured logging to make it machine-parseable.
class LogMsg(object):
    def __init__(self, text, prediction):
        self.text = text
        self.prediction = prediction
        self.time = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')

    def __str__(self):
        return json.dumps({'time': self.time, 'text': self.text, 'prediction': self.prediction})

# Webserver
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['POST'])
def get_prediction():
    comment_text = request.json['comment']
    prediction = predict_label(comment_text)
    app.logger.info(LogMsg(comment_text, prediction))
    return "\"" + comment_text + "\"" + " = " + prediction
