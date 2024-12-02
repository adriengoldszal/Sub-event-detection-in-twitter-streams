import pandas as pd
import numpy as pd
import tensorflow as tf
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Data preprocessing
def clean_tweet(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower().strip()
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def create_features(df):
    # Group by ID to handle multiple tweets per minute
    grouped = df.groupby('ID')
    
    features = pd.DataFrame()
    features['tweet_count'] = grouped['Tweet'].count()
    features['avg_tweet_length'] = grouped['Tweet'].apply(lambda x: x.str.len().mean())
    features['period_id'] = grouped['PeriodID'].first()
    features['event'] = grouped['EventID'].first()
    
    return features

def create_sequences(features, sequence_length=5):
    sequences = []
    labels = []
    
    # Sort by PeriodID to maintain temporal order
    features = features.sort_values('period_id')
    
    for i in range(len(features) - sequence_length):
        seq = features.iloc[i:i+sequence_length]
        label = features.iloc[i+sequence_length]['event']
        
        sequences.append(seq.values)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

# Model creation
def create_model(sequence_length, n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, 
                           input_shape=(sequence_length, n_features)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def main():
    # Load data
    df = pd.read_csv('worldcup_tweets.csv')
    
    # Clean tweets
    df['Tweet'] = df['Tweet'].apply(clean_tweet)
    
    # Create features
    features_df = create_features(df)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['tweet_count', 'avg_tweet_length']
    features_df[numerical_cols] = scaler.fit_transform(features_df[numerical_cols])
    
    # Create sequences
    X, y = create_sequences(features_df)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Create and train model
    model = create_model(sequence_length=5, n_features=X.shape[2])
    
    # Handle class imbalance
    class_weights = {0: 1, 1: len(y_train[y_train==0]) / len(y_train[y_train==1])}
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        class_weight=class_weights
    )
    
    return model, scaler

if __name__ == "__main__":
    main()