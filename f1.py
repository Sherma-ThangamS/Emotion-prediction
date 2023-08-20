from flask import Flask, request, render_template
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import re
import json
import gzip
# Load the saved model
model = tf.keras.models.load_model("emotion_model_learning.h5")
data=[]
lable=[]
max_len = 50
with gzip.open("train.jsonl.gz") as f:
    for l in f:
        j=json.loads(l.decode('utf-8'))
        data.append(j['text'])
        lable.append(j['label'])
app = Flask(__name__, template_folder='C:/_PROJECT_/Emotion')
def get_sequences(tokenizer,tweets):
    sequences=tokenizer.texts_to_sequences(tweets)
    padded=pad_sequences(sequences,truncating='post',padding='post',maxlen=max_len)
    padded=np.squeeze(padded)
    return padded
# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text-input']
    tokenizer=Tokenizer(num_words=10000,oov_token='<UNK>')
    tokenizer.fit_on_texts(data)
    # input_sequence = tokenizer.texts_to_sequences([text])
    # input_padded = pad_sequences(input_sequence, maxlen=max_len, padding='post',truncating='post')
    # if len(input_padded[0]) < max_len:
    #     input_padded = input_padded[:, :len(input_padded[0])]
    index_to_class={ 5:'surprise',1:'joy',0:'sadness',3:'anger',4:'fear',2:'love'}
    input_padded=get_sequences(tokenizer,[text])
    p=model.predict(np.expand_dims(input_padded,axis=0))[0]
    # p=model.predict(eval_tweets[i])
    pred_class=index_to_class[np.argmax(p).astype('uint8')]
    # emotions = list(data.sentiment.unique())
    # emotion_label = emotions[np.argmax(prediction)]
    # emotion_label=index_to_class[int(prediction.max())]
    # Return the predicted emotion label
    return render_template("index.html", prediction_text='The predicted emotion is {}'.format(pred_class))

if __name__ == '__main__':
    app.run(debug=True)
