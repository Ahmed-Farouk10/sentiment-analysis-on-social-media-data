import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.utils import to_categorical

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_nn_model(model, X_val, y_val, X_test, twitter_df):
    """Evaluate neural network model."""
    try:
        y_pred_probs = model.predict(X_val)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_val_classes = np.argmax(y_val, axis=1)
        
        print("Neural Network Validation Results:")
        print(classification_report(y_val_classes, y_pred))
        logger.info("NN classification report generated")

        cm = confusion_matrix(y_val_classes, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('NN Confusion Matrix')
        plt.savefig('nn_confusion_matrix.png')
        plt.close()
        logger.info("NN confusion matrix saved")

        twitter_pred_probs = model.predict(X_test)
        twitter_pred = np.argmax(twitter_pred_probs, axis=1)
        twitter_pred = pd.Series(twitter_pred).map({0: -1, 1: 0, 2: 1}).values
        twitter_df['predicted_sentiment_nn'] = twitter_pred
        twitter_df.to_csv('data/Twitter_Predictions_NN.csv', index=False)
        logger.info("NN Twitter predictions saved")
    except Exception as e:
        logger.error(f"Error in NN evaluation: {str(e)}")
        raise

def evaluate_tfidf_model(model, X_val, y_val, X_test, twitter_df, vectorizer):
    """Evaluate TF-IDF model."""
    try:
        y_pred = model.predict(X_val)
        
        print("TF-IDF Validation Results:")
        print(classification_report(y_val, y_pred))
        logger.info("TF-IDF classification report generated")

        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('TF-IDF Confusion Matrix')
        plt.savefig('tfidf_confusion_matrix.png')
        plt.close()
        logger.info("TF-IDF confusion matrix saved")

        twitter_pred = model.predict(X_test)
        twitter_df['predicted_sentiment_tfidf'] = twitter_pred
        twitter_df.to_csv('data/Twitter_Predictions_TFIDF.csv', index=False)
        logger.info("TF-IDF Twitter predictions saved")
    except Exception as e:
        logger.error(f"Error in TF-IDF evaluation: {str(e)}")
        raise

def main():
    """Main evaluation pipeline."""
    logger.info("Starting evaluation pipeline")

    # Load data
    X_val_nn = np.load('data/X_reddit_nn.npy')[int(0.8 * 37000):]  # Assuming 37k rows
    X_test_nn = np.load('data/X_twitter_nn.npy')
    y_val_nn = to_categorical(pd.read_csv('data/Reddit_Cleaned.csv')['category'].map({-1: 0, 0: 1, 1: 2}).values[int(0.8 * 37000):], num_classes=3)
    X_val_tfidf = np.load('data/X_reddit_tfidf.npy')[int(0.8 * 37000):]
    X_test_tfidf = np.load('data/X_twitter_tfidf.npy')
    y_val_tfidf = pd.read_csv('data/Reddit_Cleaned.csv')['category'].map({-1: 0, 0: 1, 1: 2}).values[int(0.8 * 37000):]
    twitter_df = pd.read_csv('data/Twitter_Cleaned.csv')

    # Load models and vectorizer
    nn_model = load_model('data/nn_model.h5')
    with open('data/tfidf_model.pkl', 'rb') as f:
        tfidf_model = pickle.load(f)
    with open('data/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    # Evaluate models
    evaluate_nn_model(nn_model, X_val_nn, y_val_nn, X_test_nn, twitter_df)
    evaluate_tfidf_model(tfidf_model, X_val_tfidf, y_val_tfidf, X_test_tfidf, twitter_df, tfidf_vectorizer)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise