label_names = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 
    'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

chosen_indices = [0, 1, 2, 3, 7, 10, 15, 17, 18, 20, 24, 25, 26, 27]

import random

def predict_label(comment):
    return label_names[random.choice(chosen_indices)]

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

from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['POST'])
def get_prediction():
    comment_text = request.json['comment']
    prediction = predict_label(comment_text)
    app.logger.info(LogMsg(comment_text, prediction))
    return "\"" + comment_text + "\"" + " = " + prediction