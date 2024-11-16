import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
# Load GloVe model with Gensim's API
embeddings_start_time = time.time()
print(f"Loading embeddings...")
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings
print(f"Loading embeddings took {time.time() - embeddings_start_time :.2f} seconds")

# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


def batch_preprocess_tweets(df, batch_size=1000):
    """Main preprocessing function with filtering and batching
    Link here https://www.lix.polytechnique.fr/~nikolentzos/files/meladianos_ecir18
    
        1) Removing retweets
        2) Removing duplicates
        3) Removing @ mentions
    
    """
    
    print("Starting tweet preprocessing...")
    total_start = time.time()
    
    # Create a copy to avoid modifying original
    processed_df = df.copy()
    
    # Initial data filtering
    print("\nFiltering tweets...")
    initial_count = len(processed_df)
    
    # 1. Remove retweets
    processed_df = processed_df[~processed_df['Tweet'].str.startswith('RT ', na=False)]
    retweets_removed = initial_count - len(processed_df)
    
    # 2. Remove duplicates
    processed_df = processed_df.drop_duplicates(subset=['Tweet'])
    duplicates_removed = initial_count - retweets_removed - len(processed_df)
    
    # 3. Remove tweets with @-mentions
    processed_df = processed_df[~processed_df['Tweet'].str.contains('@', na=False)]
    mentions_removed = initial_count - retweets_removed - duplicates_removed - len(processed_df)
    
    # Print filtering statistics
    print(f"Removed {retweets_removed} retweets")
    print(f"Removed {duplicates_removed} duplicates")
    print(f"Removed {mentions_removed} tweets with @-mentions")
    print(f"Remaining tweets: {len(processed_df)}")
    
    # Handle any remaining NaN values
    processed_df['Tweet'] = processed_df['Tweet'].fillna('')
    
    # Calculate number of batches
    n_batches = int(np.ceil(len(processed_df) / batch_size))
    
    # Process in batches with progress bar
    processed_tweets = []
    with tqdm(total=len(processed_df), desc="Processing tweets") as pbar:
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(processed_df))
            
            # Get current batch
            batch = processed_df['Tweet'].iloc[start_idx:end_idx]
            
            # Process batch
            batch_results = [preprocess_text(tweet) for tweet in batch]
            processed_tweets.extend(batch_results)
            
            # Update progress bar
            pbar.update(end_idx - start_idx)
    
    # Add processed tweets to DataFrame
    processed_df['Tweet'] = processed_tweets
    
    # Print timing statistics
    total_time = time.time() - total_start
    print(f"\nPreprocessing complete!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per tweet: {total_time/len(processed_df):.4f} seconds")
    
    return processed_df

def preprocess_text(text):
    """
    Performs standard text preprocessing tasks:
    1. Tokenization
    2. Stopword removal
    3. Punctuation and special character removal
    4. URL removal
    5. Porter stemming
    
    Args:
        text: String containing the tweet text
    Returns:
        Preprocessed text string
    """
    if pd.isna(text):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization : better tokenization through word_tokenize by NLTK
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization is kept (porter stemming less precise)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    return ' '.join(tokens)


# Read all training files and concatenate them into one dataframe
li = []
for filename in os.listdir("challenge_data/train_tweets"):
    df = pd.read_csv("challenge_data/train_tweets/" + filename)
    li.append(df)
df = pd.concat(li, ignore_index=True)

print(f"Number of tweets: {len(df)}")

# Apply preprocessing to each tweet
print(f"Preprocessing tweets...")
tweet_processing_start = time.time()
df = batch_preprocess_tweets(df)
print(f"Preprocessing took {time.time() - tweet_processing_start:.2f} seconds")
print(df.head(300))

vector_size = 200  # Adjust based on the chosen GloVe model
print(f"Computing tweet embeddings...")
embedding_start = time.time()
tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])
print(f"Embedding computation took {time.time() - embedding_start:.2f} seconds")
tweet_df = pd.DataFrame(tweet_vectors)
print(tweet_df.head(300))

# Attach the vectors into the original dataframe
period_features = pd.concat([df, tweet_df], axis=1)
# Drop the columns that are not useful anymore
period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
# Group the tweets into their corresponding periods. This way we generate an average embedding vector for each period
period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

# We drop the non-numerical features and keep the embeddings values for each period
X = period_features.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
# We extract the labels of our training samples
y = period_features['EventType'].values

###### Evaluating on a test set:

# We split our data into a training and test set that we can use to train our classifier without fine-tuning into the
# validation set and without submitting too many times into Kaggle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# We set up a basic classifier that we train and then calculate the accuracy on our test set
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set: ", accuracy_score(y_test, y_pred))

###### For Kaggle submission

# This time we train our classifier on the full dataset that it is available to us.
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)
# We add a dummy classifier for sanity purposes
dummy_clf = DummyClassifier(strategy="most_frequent").fit(X, y)

predictions = []
dummy_predictions = []
# We read each file separately, we preprocess the tweets and then use the classifier to predict the labels.
# Finally, we concatenate all predictions into a list that will eventually be concatenated and exported
# to be submitted on Kaggle.
for fname in os.listdir("challenge_data/eval_tweets"):
    val_df = pd.read_csv("challenge_data/eval_tweets/" + fname)
    val_df = batch_preprocess_tweets(val_df)
    val_df.head(300)

    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in val_df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    period_features = pd.concat([val_df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    X = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    preds = clf.predict(X)
    dummy_preds = dummy_clf.predict(X)

    period_features['EventType'] = preds
    period_features['DummyEventType'] = dummy_preds

    predictions.append(period_features[['ID', 'EventType']])
    dummy_predictions.append(period_features[['ID', 'DummyEventType']])

pred_df = pd.concat(predictions)
pred_df.to_csv('logistic_better_preprocessing_predictions.csv', index=False)

pred_df = pd.concat(dummy_predictions)
pred_df.to_csv('dummy_better_preprocessing_predictions.csv', index=False)

