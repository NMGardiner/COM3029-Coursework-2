# Model
from transformers import AutoTokenizer, BertForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("model")
model = BertForSequenceClassification.from_pretrained("model")

labels = ["admiration", "amusement", "anger", "annoyance", "curiosity", "disapproval", "gratitude", "joy", "love", "optimism", "remorse", "sadness", "surprise", "neutral"]

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
    
class LogMsgInputFormat400Error(object):
    def __init__(self):
        self.time = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')

    def __str__(self):
        return json.dumps({'time': self.time, 'error': "The message must be JSON in the form json={'comment': string_to_predict}."})

# Webserver
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def get_prediction():
    if request.method == 'GET':
        return "To receive predictions, send a POST request in the form json={'comment': string_to_predict}."
    else:
        if ((type(request.json) != dict) or (list(request.json.keys()) != ['comment']) or (type(request.json['comment']) != str) or (len(request.json["comment"])==0)):
            app.logger.info(LogMsgInputFormat400Error())
            return json.dumps({"success": False, "error":"The message must be JSON in the form json={'comment': string_to_predict}."}), 400
        comment_text = request.json['comment']
        prediction = predict_label(comment_text)
        app.logger.info(LogMsg(comment_text, prediction))
        return json.dumps({"success": True, "prediction": prediction})