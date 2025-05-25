import pandas as pd
import spacy
import logging
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Set data directory path
DATA_DIR = 'data'

def preprocess_text_layers(text):
    """Layered preprocessing for neural network and TF-IDF."""
    # Handle NaN and empty strings silently
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return ''
    
    # Layer 1: Basic Text Cleaning
    text = text.lower().strip()
    # Remove URLs, special characters, and extra whitespace
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Skip spaCy processing if text is too short
    if len(text.split()) < 2:
        return text
        
    # Layer 2: Lemmatization and Stop Word Removal
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    processed_text = ' '.join(tokens)
    
    # Return empty string if nothing left after processing
    return processed_text if processed_text.strip() else ''

def preprocess_for_nn(reddit_texts, twitter_texts, max_words=5000, max_len=100):
    """Prepare sequences for neural network."""
    try:
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(reddit_texts)
        logger.info(f"Tokenizer fitted with {len(tokenizer.word_index)} unique words")

        X_reddit = tokenizer.texts_to_sequences(reddit_texts)
        X_twitter = tokenizer.texts_to_sequences(twitter_texts)
        
        X_reddit = pad_sequences(X_reddit, maxlen=max_len)
        X_twitter = pad_sequences(X_twitter, maxlen=max_len)
        logger.info(f"NN sequences prepared: Reddit shape {X_reddit.shape}, Twitter shape {X_twitter.shape}")
        
        return X_reddit, X_twitter, tokenizer
    except Exception as e:
        logger.error(f"Error in NN preprocessing: {str(e)}")
        raise

def preprocess_for_tfidf(reddit_texts, twitter_texts):
    """Prepare TF-IDF features."""
    try:
        vectorizer = TfidfVectorizer(max_features=5000)
        X_reddit = vectorizer.fit_transform(reddit_texts)
        X_twitter = vectorizer.transform(twitter_texts)
        logger.info(f"TF-IDF features prepared: Reddit shape {X_reddit.shape}, Twitter shape {X_twitter.shape}")
        
        # Save vectorizer for later use
        with open(os.path.join(DATA_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
            import pickle
            pickle.dump(vectorizer, f)
        logger.info("TF-IDF vectorizer saved")
        
        return X_reddit, X_twitter
    except Exception as e:
        logger.error(f"Error in TF-IDF preprocessing: {str(e)}")
        raise

def main():
    """Main preprocessing pipeline."""
    logger.info("Starting preprocessing pipeline")

    # Load data
    try:
        reddit_df = pd.read_csv(os.path.join(DATA_DIR, 'Reddit_Data.csv'))
        twitter_df = pd.read_csv(os.path.join(DATA_DIR, 'Twitter_Data.csv'))
        logger.info(f"Reddit data loaded: {reddit_df.shape}")
        logger.info(f"Twitter data loaded: {twitter_df.shape}")
        logger.info(f"Reddit missing values: {reddit_df.isnull().sum()}")
        logger.info(f"Twitter missing values: {twitter_df.isnull().sum()}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

    # Preprocess text in batches
    logger.info("Preprocessing text data")
    batch_size = 1000
    
    # Process Reddit data in batches
    reddit_cleaned = []
    for i in range(0, len(reddit_df), batch_size):
        batch = reddit_df['clean_comment'].iloc[i:i+batch_size]
        cleaned_batch = batch.apply(preprocess_text_layers)
        reddit_cleaned.extend(cleaned_batch)
        logger.info(f"Processed Reddit batch {i//batch_size + 1}/{(len(reddit_df)-1)//batch_size + 1}")
    
    reddit_df['cleaned'] = reddit_cleaned
    
    # Process Twitter data in batches
    twitter_cleaned = []
    for i in range(0, len(twitter_df), batch_size):
        batch = twitter_df['clean_text'].iloc[i:i+batch_size]
        cleaned_batch = batch.apply(preprocess_text_layers)
        twitter_cleaned.extend(cleaned_batch)
        logger.info(f"Processed Twitter batch {i//batch_size + 1}/{(len(twitter_df)-1)//batch_size + 1}")
    
    twitter_df['cleaned'] = twitter_cleaned

    # Save cleaned data
    logger.info("Saving cleaned data...")
    reddit_df.to_csv(os.path.join(DATA_DIR, 'Reddit_Cleaned.csv'), index=False)
    twitter_df.to_csv(os.path.join(DATA_DIR, 'Twitter_Cleaned.csv'), index=False)
    logger.info("Cleaned data saved")

    # Prepare data for neural network
    logger.info("Preparing neural network data...")
    X_reddit_nn, X_twitter_nn, tokenizer = preprocess_for_nn(reddit_df['cleaned'], twitter_df['cleaned'])
    np.save(os.path.join(DATA_DIR, 'X_reddit_nn.npy'), X_reddit_nn)
    np.save(os.path.join(DATA_DIR, 'X_twitter_nn.npy'), X_twitter_nn)
    with open(os.path.join(DATA_DIR, 'tokenizer.pkl'), 'wb') as f:
        import pickle
        pickle.dump(tokenizer, f)
    logger.info("Neural network data saved")

    # Prepare data for TF-IDF
    logger.info("Preparing TF-IDF data...")
    X_reddit_tfidf, X_twitter_tfidf = preprocess_for_tfidf(reddit_df['cleaned'], twitter_df['cleaned'])
    np.save(os.path.join(DATA_DIR, 'X_reddit_tfidf.npy'), X_reddit_tfidf.toarray())
    np.save(os.path.join(DATA_DIR, 'X_twitter_tfidf.npy'), X_twitter_tfidf.toarray())
    logger.info("TF-IDF data saved")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise