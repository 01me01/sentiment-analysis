from flask import Flask, render_template,flash,request,url_for
import numpy as np
import pandas as pd
import re
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')


img_folder = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['Upload folder'] = img_folder

def init():
    global model,graph

    model = load_model('sentiment_analysis_model_new.h5')
    graph = tf.compat.v1.get_default_graph()


@app.route('/',methods = ['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/analysis_pred', methods = ['POST', "GET"])
def analysis_pred():
    # if request.method=='POST':
    text = request.form.get('text')

    sid = SentimentIntensityAnalyzer()
    if text is not None:
        score = sid.polarity_scores(text)
        if score['compound'] < 0:
            Sentiment = 'Negative'
            img_filename = os.path.join(app.config['Upload folder'], 'Sad_Emoji.png')
        elif (score['compound']==0 or (score['neu']>score['pos'])):
            Sentiment ='Neutral'
            img_filename = os.path.join(app.config['Upload folder'], 'neutral_emoji.png')
            print("neutral")
        else:
            Sentiment = 'Positive'
            img_filename = os.path.join(app.config['Upload folder'], 'Smiling_Emoji.png')
            print("positive")
    else:
    # Handle the case where text is None
        Sentiment = 'Unknown'  # Or any other appropriate value
        img_filename = ''  # Or h
        score=0
    return render_template('analysis.html', text=text, sentiment=Sentiment, probability=score, image=img_filename)
#########################Code for Sentiment Analysis

if __name__ == "__main__":
    init()
    app.debug=True
    app.run()