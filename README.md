# Social Media Sentiment Analysis

This project performs sentiment analysis on social media data from Reddit and Twitter using both Neural Network and TF-IDF approaches.

## Project Structure

```
sentiment-analysis-on-social-media-data/
├── data/                      # Data directory
│   ├── Reddit_Data.csv       # Raw Reddit data
│   ├── Twitter_Data.csv      # Raw Twitter data
│   ├── Reddit_Cleaned.csv    # Preprocessed Reddit data
│   ├── Twitter_Cleaned.csv   # Preprocessed Twitter data
│   └── [other processed files]
├── preprocess.py             # Data preprocessing pipeline
├── train.py                  # Model training pipeline
├── evaluate.py              # Model evaluation pipeline
└── requirements.txt         # Project dependencies
```

## Data Preprocessing (`preprocess.py`)

The preprocessing pipeline handles text cleaning and feature extraction:

1. **Text Cleaning**:
   - Converts text to lowercase
   - Removes URLs and special characters
   - Removes extra whitespace
   - Handles empty/invalid text

2. **Feature Extraction**:
   - Neural Network features: Text tokenization and sequence padding
   - TF-IDF features: Term frequency-inverse document frequency

3. **Output Files**:
   - `Reddit_Cleaned.csv`: Cleaned Reddit data
   - `Twitter_Cleaned.csv`: Cleaned Twitter data
   - `X_reddit_nn.npy`: Neural network features for Reddit
   - `X_twitter_nn.npy`: Neural network features for Twitter
   - `X_reddit_tfidf.npy`: TF-IDF features for Reddit
   - `X_twitter_tfidf.npy`: TF-IDF features for Twitter

## Model Training (`train.py`)

Two models are trained:

1. **Neural Network Model**:
   - Architecture: Embedding → LSTM → Dropout → Dense layers
   - Input: Tokenized and padded text sequences
   - Output: 3-class sentiment prediction (-1, 0, 1)
   - Saved as: `nn_model.h5`

2. **TF-IDF Model**:
   - Algorithm: Logistic Regression
   - Input: TF-IDF features
   - Output: 3-class sentiment prediction
   - Saved as: `tfidf_model.pkl`

## Model Evaluation (`evaluate.py`)

Evaluates both models on validation and test data:

1. **Metrics**:
   - Classification report (precision, recall, F1-score)
   - Confusion matrix visualization

2. **Output Files**:
   - `nn_confusion_matrix.png`: Neural network confusion matrix
   - `tfidf_confusion_matrix.png`: TF-IDF confusion matrix
   - `Twitter_Predictions_NN.csv`: Twitter predictions using NN
   - `Twitter_Predictions_TFIDF.csv`: Twitter predictions using TF-IDF

## Setup and Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline**:
   ```bash
   # Preprocess data
   python preprocess.py
   
   # Train models
   python train.py
   
   # Evaluate models
   python evaluate.py
   ```

## Dependencies

- Python 3.x
- pandas
- numpy
- tensorflow
- scikit-learn
- spacy
- matplotlib
- seaborn

## Data Format

1. **Reddit Data**:
   - `clean_comment`: Text content
   - `category`: Sentiment label (-1, 0, 1)

2. **Twitter Data**:
   - `clean_text`: Text content
   - `category`: Sentiment label (-1, 0, 1)

## Model Performance

The project evaluates both models on:
- Validation set (20% of Reddit data)
- Test set (Twitter data)

Performance metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

## Notes

- The preprocessing handles missing values and empty text
- Models are trained on Reddit data and tested on Twitter data
- Both models support 3-class sentiment classification
- Results are saved in both CSV and visualization formats
