from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import re
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the saved models and scalers
def load_models():
    try:
        log_reg_hybrid = pickle.load(open('log_reg_hybrid_model.pkl', 'rb'))
        tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return log_reg_hybrid, tfidf_vectorizer, scaler
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return None, None, None

# Load the BERT tokenizer and model
def load_bert_model():
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        return tokenizer, distilbert_model
    except Exception as e:
        logging.error(f"Error loading BERT model: {e}")
        return None, None

# Determine the device (CPU or GPU)
def get_device():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    except Exception as e:
        logging.error(f"Error determining device: {e}")
        return None

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function for input tweets
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z@\s#]', '', text)  # Keep only letters, spaces, '@', and '#'
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to get BERT embeddings for a single tweet
def get_bert_embeddings(text, tokenizer, distilbert_model, device):
    try:
        inputs = tokenizer([text], return_tensors='pt', truncation=True, padding=True, max_length=64).to(device)
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token representation
    except Exception as e:
        logging.error(f"Error getting BERT embeddings: {e}")
        return None

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# API route to receive the tweets and return predictions
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    # Load models and BERT tokenizer/model
    log_reg_hybrid, tfidf_vectorizer, scaler = load_models()
    tokenizer, distilbert_model = load_bert_model()
    device = get_device()

    if not log_reg_hybrid or not tokenizer or not device:
        return jsonify({'error': 'Model or device not loaded properly.'}), 500

    # Check if a CSV file was uploaded
    if 'csv_file' in request.files and request.files['csv_file'].filename != '':
        file = request.files['csv_file']
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file)
        if 'tweet' not in df.columns:
            return jsonify({'error': 'CSV must contain a "tweet" column.'}), 400
        tweets_list = df['tweet'].dropna().tolist()  # Extract tweets from the "tweet" column

    # Check if tweets were entered in the textarea
    elif 'tweets' in request.form and request.form['tweets'].strip():
        tweets_input = request.form.get('tweets', '')
        tweets_list = [tweet.strip() for tweet in tweets_input.strip().split('\n') if tweet.strip()]
    
    else:
        return jsonify({'error': 'No input provided. Please either upload a CSV file or enter tweets.'}), 400

    # Get names/entities if any were provided
    names_input = request.form.get('names', '')
    names_list = [name.strip().lower() for name in names_input.strip().split(',') if name.strip()]

    # Initialize data structures to hold sentiments and entity-based results
    sentiments = []
    entity_sentiments = {name: {'Positive': 0, 'Negative': 0} for name in names_list}

    # Iterate over the list of tweets
    for tweet in tweets_list:
        # Preprocess the tweet (cleaning)
        preprocessed_tweet = preprocess_text(tweet)

        # Get TF-IDF features for the tweet
        tfidf_features = tfidf_vectorizer.transform([preprocessed_tweet]).toarray()

        # Get BERT embeddings for the tweet
        bert_embedding = get_bert_embeddings(preprocessed_tweet, tokenizer, distilbert_model, device)
        if bert_embedding is None:
            continue  # Skip if embedding failed

        # Additional feature engineering (length of tweet, punctuation counts)
        tweet_length = np.array([[len(preprocessed_tweet.split())]])  # Length of the tweet (in words)
        num_exclamations = np.array([[preprocessed_tweet.count('!')]])  # Count of exclamation marks
        num_questions = np.array([[preprocessed_tweet.count('?')]])  # Count of question marks

        # Scale additional features
        additional_features = np.hstack([tweet_length, num_exclamations, num_questions])
        additional_features_scaled = scaler.transform(additional_features)

        # Combine TF-IDF, BERT embeddings, and additional features into one feature vector
        combined_features = np.hstack((tfidf_features, bert_embedding, additional_features_scaled))

        # Use the hybrid logistic regression model to predict sentiment
        prediction = log_reg_hybrid.predict(combined_features)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        sentiments.append({'tweet': tweet, 'sentiment': sentiment})

        # Check for entity mentions and record their sentiment
        for name in names_list:
            if name in preprocessed_tweet.lower():
                entity_sentiments[name][sentiment] += 1

    # Summarize sentiment counts (for display purposes)
    sentiment_counts = Counter([s['sentiment'] for s in sentiments])
    sentiment_data = {
        'Positive': sentiment_counts.get('Positive', 0),
        'Negative': sentiment_counts.get('Negative', 0)
    }

    # Prepare entity sentiment data for chart display
    entity_sentiments_labels = list(entity_sentiments.keys())
    entity_sentiments_data = [entity_sentiments[name]['Positive'] - entity_sentiments[name]['Negative'] for name in entity_sentiments_labels]

    # Render the result.html template with the prediction results
    return render_template('result.html', 
                           sentiments=sentiments, 
                           sentiment_data=sentiment_data, 
                           entity_sentiments_labels=entity_sentiments_labels, 
                           entity_sentiments_data=entity_sentiments_data)

# Run the app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
