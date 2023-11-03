import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the comments from the CSV file
comments_df = pd.read_csv('youtube_comments.csv')
comments = comments_df['Comment'].values

# Tokenize the comments
tokenizer = Tokenizer(num_words=10000)  # You can adjust the vocabulary size
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
padded_sequences = pad_sequences(sequences, maxlen=100)  # You can adjust the sequence length

# Load the pre-trained RNN model
model = load_model('sentiment_model.h5')  # Load your saved model here

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        comment = request.form['comment']
        sequence = tokenizer.texts_to_sequences([comment])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        prediction = model.predict(padded_sequence)
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        return render_template('index.html', comment=comment, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
