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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import sys
from transformers import AutoTokenizer, AutoModel
import torch
print(sys.executable)


### We will compare three types of preprocessing : Glove 200, Glove 50 and BERT

### Glove
### Defining some functions that will help preprocess with Glove models

# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=50):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if (
        not word_vectors
    ):  # If no words in the tweet are in the vocabulary, return a zero vector
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
            batch_results = [preprocess_text(tweet) for tweet in batch]
            processed_tweets.extend(batch_results)

            # Update progress bar
            pbar.update(end_idx - start_idx)

    # Add processed tweets to DataFrame
    processed_df["Tweet"] = processed_tweets

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

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Tokenization : better tokenization through word_tokenize by NLTK
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization is kept (porter stemming less precise)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into text
    return " ".join(tokens)


# Read all training files and concatenate them into one dataframe
li = []
for filename in os.listdir("challenge_data/train_tweets"):
    df = pd.read_csv("challenge_data/train_tweets/" + filename)
    li.append(df)
df = pd.concat(li, ignore_index=True)

print(f"Number of tweets: {len(df)}")

# Apply preprocessing to each training tweet
print(f"Preprocessing training tweets...")
tweet_processing_start = time.time()
df = batch_preprocess_tweets(df)
print(f"Preprocessing took {time.time() - tweet_processing_start:.2f} seconds")
print(df.head(300))
df.to_csv("preprocessed-data/preprocessed_tweets.csv", index=False)

original_count = len(df)
df = df.dropna() 
rows_dropped = original_count - len(df)
print(f"Rows dropped {rows_dropped}")

# Read all eval files and concatenate them into one dataframe
li = []
for filename in os.listdir("challenge_data/eval_tweets"):
    eval_df = pd.read_csv("challenge_data/eval_tweets/" + filename)
    li.append(eval_df)
eval_df = pd.concat(li, ignore_index=True)

print(f"Number of tweets: {len(eval_df)}")

# Apply preprocessing to each evaluation tweet
print(f"Preprocessing evaluation tweets...")
tweet_processing_start = time.time()
eval_df = batch_preprocess_tweets(eval_df)
print(f"Preprocessing took {time.time() - tweet_processing_start:.2f} seconds")
eval_df.to_csv("preprocessed-data/preprocessed_tweets_test_no_retweets_etc.csv", index=False)

### Glove 200
print("Starting GloVe 200...")

# Download some NLP models for processing, optional
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")
# Load GloVe model with Gensim's API
embeddings_start_time = time.time()
print(f"Loading embeddings...")
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings
print(f"Loading embeddings took {time.time() - embeddings_start_time :.2f} seconds")

vector_size = 200  # Adjust based on the chosen GloVe model
print(f"Computing training tweet embeddings...")
embedding_start = time.time()
tweet_vectors = np.vstack(
    [get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df["Tweet"]]
)
print(f"Embedding computation took {time.time() - embedding_start:.2f} seconds")
tweet_df = pd.DataFrame(tweet_vectors)

# Attach the vectors into the original dataframe
period_features = pd.concat([df.reset_index(drop=True), tweet_df], axis=1)
# Drop the columns that are not useful anymore
period_features = period_features.drop(columns=["Timestamp", "Tweet"])
# Group the tweets into their corresponding periods. This way we generate an average embedding vector for each period
period_features = (
    period_features.groupby(["MatchID", "PeriodID", "ID"]).mean().reset_index()
)
period_features.to_csv("preprocessed-data/period_features_glove_200.csv", index=False)

# eval data
vector_size = 200  # Adjust based on the chosen GloVe model
print(f"Computing evaluation tweet embeddings...")
# Compute embeddings for the concatenated evaluation data
tweet_vectors_test = np.vstack(
    [get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in eval_df["Tweet"]]
)
tweet_df_test = pd.DataFrame(tweet_vectors_test)

# Attach the vectors into the original dataframe
period_features_test = pd.concat([eval_df.reset_index(drop=True), tweet_df_test], axis=1)
period_features_test = period_features_test.drop(columns=["Timestamp", "Tweet"])
# Group the tweets into their corresponding periods
period_features_test = (
    period_features_test.groupby(["MatchID", "PeriodID", "ID"]).mean().reset_index()
)
print(period_features_test.head())
period_features_test.to_csv("preprocessed-data/period_features_test_glove_200_no_retweets_etc.csv", index=False)
print("GloVe 200 done.")

### Glove 50
print("Starting GloVe 50...")
# Download some NLP models for processing, optional
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")
# Load GloVe model with Gensim's API
embeddings_start_time = time.time()
print(f"Loading embeddings...")
embeddings_model = api.load("glove-twitter-50")  # 50-dimensional GloVe embeddings
print(f"Loading embeddings took {time.time() - embeddings_start_time :.2f} seconds")

vector_size = 50  # Adjust based on the chosen GloVe model
print(f"Computing training tweet embeddings...")
embedding_start = time.time()
tweet_vectors = np.vstack(
    [get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df["Tweet"]]
)
print(f"Embedding computation took {time.time() - embedding_start:.2f} seconds")
tweet_df = pd.DataFrame(tweet_vectors)

# Attach the vectors into the original dataframe
period_features = pd.concat([df.reset_index(drop=True), tweet_df], axis=1)
# Drop the columns that are not useful anymore
period_features = period_features.drop(columns=["Timestamp", "Tweet"])
# Group the tweets into their corresponding periods. This way we generate an average embedding vector for each period
period_features = (
    period_features.groupby(["MatchID", "PeriodID", "ID"]).mean().reset_index()
)
period_features.to_csv("preprocessed-data/period_features_glove_50.csv", index=False)

# eval data
vector_size = 50  # Adjust based on the chosen GloVe model
print(f"Computing evaluation tweet embeddings...")
# Compute embeddings for the concatenated evaluation data
tweet_vectors_test = np.vstack(
    [get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in eval_df["Tweet"]]
)
tweet_df_test = pd.DataFrame(tweet_vectors_test)

# Attach the vectors into the original dataframe
period_features_test = pd.concat([eval_df.reset_index(drop=True), tweet_df_test], axis=1)
period_features_test = period_features_test.drop(columns=["Timestamp", "Tweet"])
# Group the tweets into their corresponding periods
period_features_test = (
    period_features_test.groupby(["MatchID", "PeriodID", "ID"]).mean().reset_index()
)
print(period_features_test.head())
period_features_test.to_csv("preprocessed-data/period_features_test_glove_50_no_retweets_etc.csv", index=False)
print("GloVe 50 done.")

### BERT
print("Starting BERT...")
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
            batch_results = [preprocess_text(tweet) for tweet in batch]
            processed_tweets.extend(batch_results)

            # Update progress bar
            pbar.update(end_idx - start_idx)

    # Add processed tweets to DataFrame
    processed_df["Tweet"] = processed_tweets

    # Print timing statistics
    total_time = time.time() - total_start
    print(f"\nPreprocessing complete!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per tweet: {total_time/len(processed_df):.4f} seconds")

    return processed_df


def preprocess_text(text):
    """
    Limited preprocessing for BERT

    Args:
        text: String containing the tweet text
    Returns:
        Preprocessed text string
    """

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    text = " ".join(text.split())

    # Join tokens back into text
    return text

# Read all training files and concatenate them into one dataframe
li = []
for filename in os.listdir("/challenge_data/train_tweets"):
    df = pd.read_csv("/challenge_data/train_tweets/" + filename)
    li.append(df)
df = pd.concat(li, ignore_index=True)

print(f"Number of tweets: {len(df)}")

# Apply preprocessing to each tweet
print(f"Preprocessing training tweets...")
tweet_processing_start = time.time()
df = batch_preprocess_tweets(df)
print(f"Preprocessing took {time.time() - tweet_processing_start:.2f} seconds")
print(df.head(300))
df.to_csv("preprocessed-data/preprocessed_tweets_bert.csv", index=False)

original_count = len(df)
df = df.dropna() 
rows_dropped = original_count - len(df)
print(f"Rows dropped {rows_dropped}")

# 1. Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# 2. Function to get BERT embeddings for a batch of tweets
def get_bert_embeddings(tweets, tokenizer, model, max_length=128):
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Tokenize all tweets in the batch
    encoded = tokenizer(
        tweets,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move tensors to the same device as model
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding (first token) as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return embeddings

# 3. Process your tweets in batches
def process_tweets_with_bert(df, batch_size=1000):
    import gc  # Add at top
    all_embeddings = []
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df['Tweet'].iloc[i:i + batch_size].tolist()
        batch_embeddings = get_bert_embeddings(batch, tokenizer, model)
        all_embeddings.extend(batch_embeddings)
        gc.collect()  # Clean up each batch
        
    embeddings_array = np.array(all_embeddings, dtype=np.float16)
    del all_embeddings  # Free the list
    gc.collect()
    
    embedding_cols = [f'embedding_{i}' for i in range(embeddings_array.shape[1])]
    embeddings_df = pd.DataFrame(embeddings_array, columns=embedding_cols, index=df.index)
    del embeddings_array  # Free the array
    gc.collect()
    
    return embeddings_df

print(f"Computing training tweet embeddings...")
embedding_start = time.time()

tweet_vectors = process_tweets_with_bert(df)

print(f"Embedding computation took {time.time() - embedding_start:.2f} seconds")

# Attach the vectors into the original dataframe
period_features = pd.concat([df, tweet_vectors], axis=1)
# Drop the columns that are not useful anymore
period_features = period_features.drop(columns=["Timestamp", "Tweet"])
# Group the tweets into their corresponding periods. This way we generate an average embedding vector for each period
period_features = (
    period_features.groupby(["MatchID", "PeriodID", "ID"]).mean().reset_index()
)
period_features.to_csv("preprocessed-data/period_features_bert.csv",index=False)

# eval data
li = []
for filename in os.listdir("challenge_data/eval_tweets"):
    eval_df = pd.read_csv("challenge_data/eval_tweets/" + filename)
    li.append(eval_df)
eval_df = pd.concat(li, ignore_index=True)

# Apply preprocessing to each tweet
print(f"Preprocessing evaluation tweets...")
tweet_processing_start = time.time()
eval_df = batch_preprocess_tweets(eval_df)
print(f"Preprocessing took {time.time() - tweet_processing_start:.2f} seconds")

# Compute embeddings for the concatenated evaluation data
tweet_vectors_test = process_tweets_with_bert(eval_df)
tweet_df_test = pd.DataFrame(tweet_vectors_test)

# Attach the vectors into the original dataframe
period_features_test = pd.concat([eval_df.reset_index(drop=True), tweet_df_test], axis=1)
period_features_test = period_features_test.drop(columns=["Timestamp", "Tweet"])

# Group the tweets into their corresponding periods
period_features_test = (
    period_features_test.groupby(["MatchID", "PeriodID", "ID"]).mean().reset_index()
)
print(period_features_test.head())
period_features_test.to_csv("preprocessed-data/period_features_test_bert_no_retweets_etc.csv", index=False)
