# app.py

from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load vocabulary and model
vocab = pd.read_csv('Static/vocab.txt', header=None)[0].tolist()
loaded_model = joblib.load('Model/MultinomialNB.pkl')

# Initialize the CountVectorizer with the loaded vocabulary
Cv = CountVectorizer(vocabulary=vocab)

# Function to preprocess text
def text_preprocessor(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    clear_text = [x for x in tokens if x.isalpha()]
    remove_sw = [x for x in clear_text if x not in ENGLISH_STOP_WORDS]
    joined_sw = ' '.join(remove_sw)
    tokens = word_tokenize(joined_sw)
    lemmatized_tokens = [lemmatizer.lemmatize(word=token, pos='v') for token in tokens]
    return ' '.join(lemmatized_tokens)

# Function to predict spam
def predict_spam(text):
    preprocessed_text = text_preprocessor(text)
    vectorized_text = Cv.transform([preprocessed_text])
    prediction = loaded_model.predict(vectorized_text)[0]
    return prediction

# Route to render HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = predict_spam(text)
    if prediction == 1:
        result = "This email is spam."
    else:
        result = "This email is not spam."
    return render_template('result.html', result=result)

