import os
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from tqdm import tqdm
import bisect
from cvxpy import Variable, Minimize, norm, Problem
from scipy.sparse import find
import pickle

# Download some NLP models for processing, optional
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

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
    processed_df = processed_df[~processed_df["Tweet"].str.startswith("RT ", na=False)]
    retweets_removed = initial_count - len(processed_df)

    # 2. Remove duplicates
    processed_df = processed_df.drop_duplicates(subset=["Tweet"])
    duplicates_removed = initial_count - retweets_removed - len(processed_df)

    # 3. Remove tweets with @-mentions
    processed_df = processed_df[~processed_df["Tweet"].str.contains("@", na=False)]
    mentions_removed = (
        initial_count - retweets_removed - duplicates_removed - len(processed_df)
    )

    # Print filtering statistics
    print(f"Removed {retweets_removed} retweets")
    print(f"Removed {duplicates_removed} duplicates")
    print(f"Removed {mentions_removed} tweets with @-mentions")
    print(f"Remaining tweets: {len(processed_df)}")

    
    vocabulary = set()
    # Calculate number of batches
    n_batches = int(np.ceil(len(processed_df) / batch_size))

    # Process in batches with progress bar
    processed_tweets = []
    with tqdm(total=len(processed_df), desc="Processing tweets") as pbar:
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(processed_df))

            # Get current batch
            batch = processed_df["Tweet"].iloc[start_idx:end_idx]

            # Process batch
            batch_results, batch_vocab = zip(*[preprocess_text(tweet) for tweet in batch])
            
            for words in batch_vocab:
                vocabulary.update(words)
                
            processed_tweets.extend(batch_results)

            # Update progress bar
            pbar.update(end_idx - start_idx)

    
    vocabulary = sorted(list(vocabulary))
    
    # Add processed tweets to DataFrame
    processed_df["Tweet"] = processed_tweets

    # Print timing statistics
    total_time = time.time() - total_start
    print(f"\nPreprocessing complete!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per tweet: {total_time/len(processed_df):.4f} seconds")

    return processed_df, vocabulary


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

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Simple splitting (as done in paper)
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Porter stemming (paper uses this instead of lemmatization)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Join tokens back into text
    return " ".join(tokens), tokens

def preprocess_file(filename,save_dir="graph_of_words/preprocessed_data") :
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    base_name = os.path.basename(filename).replace('.csv', '')
    save_path = os.path.join(save_dir, f"{base_name}_preprocessed.pkl")
    
    # Read match file
    df = pd.read_csv(filename)
    
    processed_df, vocabulary = batch_preprocess_tweets(df)
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'processed_df': processed_df,
            'vocabulary': vocabulary
        }, f)
    
    print(f"Saved preprocessed data to {save_path}")
    
    return processed_df, vocabulary