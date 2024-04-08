from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from logger import setup_logger

app = Flask(__name__)

# Set up logger
logger = setup_logger()

# Load vocabulary and model
vocab = pd.read_csv('Static/vocab.txt', header=None)[0].tolist()
loaded_model = joblib.load('Model/MultinomialNB.pkl')

# Initialize the CountVectorizer with the loaded vocabulary
Cv = CountVectorizer(vocabulary=vocab)

# Function to preprocess text
def text_preprocessor(text):
    logger.info("Preprocessing text...")
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    clear_text = [x for x in tokens if x.isalpha()]
    remove_sw = [x for x in clear_text if x not in ENGLISH_STOP_WORDS]
    joined_sw = ' '.join(remove_sw)
    tokens = word_tokenize(joined_sw)
    lemmatized_tokens = [lemmatizer.lemmatize(word=token, pos='v') for token in tokens]
    logger.info("Text preprocessing completed.")
    return ' '.join(lemmatized_tokens)

# Function to vectorize text
def vectorize_text(text):
    logger.info("Vectorizing text...")
    preprocessed_text = text_preprocessor(text)
    vectorized_text = Cv.transform([preprocessed_text])
    logger.info("Text vectorization completed.")
    return vectorized_text

# Function to predict spam
def predict_spam(text):
    logger.info("Predicting spam...")
    vectorized_text = vectorize_text(text)
    prediction = loaded_model.predict(vectorized_text)[0]
    logger.info("Spam prediction completed.")
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
    logger.info(f"Prediction result: {result}")
    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
