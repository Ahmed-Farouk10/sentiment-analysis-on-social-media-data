import numpy as np
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_nn_model(X_train, y_train, max_words=5000, max_len=100):
    """Train a neural network model."""
    try:
        model = Sequential([
            Embedding(max_words, 100, input_length=max_len),
            LSTM(64, return_sequences=False),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)
        logger.info("Neural network training completed")
        model.save('data/nn_model.h5')
        logger.info("Neural network model saved to 'nn_model.h5'")
        return model
    except Exception as e:
        logger.error(f"Error in NN training: {str(e)}")
        raise

def train_tfidf_model(X_train, y_train):
    """Train a Logistic Regression model with TF-IDF."""
    try:
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X_train, y_train)
        logger.info("TF-IDF model training completed")
        with open('data/tfidf_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        logger.info("TF-IDF model saved to 'tfidf_model.pkl'")
        return model
    except Exception as e:
        logger.error(f"Error in TF-IDF training: {str(e)}")
        raise

def main():
    """Main training pipeline."""
    logger.info("Starting training pipeline")

    # Load preprocessed data
    X_reddit_nn = np.load('data/X_reddit_nn.npy')
    X_reddit_tfidf = np.load('data/X_reddit_tfidf.npy')
    y_reddit = pd.read_csv('data/Reddit_Cleaned.csv')['category'].map({-1: 0, 0: 1, 1: 2}).values
    y_reddit_nn = to_categorical(y_reddit, num_classes=3)

    # Split data
    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(X_reddit_nn, y_reddit_nn, test_size=0.2, random_state=42)
    X_train_tfidf, X_val_tfidf, y_train_tfidf, y_val_tfidf = train_test_split(X_reddit_tfidf, y_reddit, test_size=0.2, random_state=42)
    logger.info(f"Training sets: NN {X_train_nn.shape}, TF-IDF {X_train_tfidf.shape}")

    # Train models
    nn_model = train_nn_model(X_train_nn, y_train_nn)
    tfidf_model = train_tfidf_model(X_train_tfidf, y_train_tfidf)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise