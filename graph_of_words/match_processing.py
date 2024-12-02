import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from scipy.sparse import find
import pickle
import gc

from preprocessing import *
from event_detection import *

def process_time_window(preprocessed_tweets, vocabulary):
    """
    Process tweets from a single time window.
    
    Args:
        preprocessed_tweets: List of preprocessed tweets
        vocabulary: Sorted list of unique words
        
    Returns:
        Dictionary containing period data
    """
    # Generate adjacency matrix
    adj_matrix, tweets_edges = generate_adjacency_matrix_dense(preprocessed_tweets, vocabulary)
    
    # Generate vector representation
    vector, vector_nodes, vector_edges, weighted_edges = generate_vector(adj_matrix, vocabulary)
    
    result = {
        'vocabulary': vocabulary,
        'adjacency_matrix': adj_matrix,
        'tweets_edges': tweets_edges,
        'vector': vector,
        'vector_nodes': vector_nodes,
        'vector_edges': vector_edges,
        'weighted_edges': weighted_edges
    }
    
    # Explicit cleanup
    del adj_matrix
    del tweets_edges
    del vector
    del vector_nodes
    del vector_edges
    del weighted_edges
    gc.collect()
    
    return result
    
def process_single_match_file(filename, save_dir = "graph_of_words/preprocessed_data"):
    
    base_name = os.path.basename(filename).replace('.csv', '')
    save_path = os.path.join(save_dir, f"{base_name}_preprocessed.pkl")
    
    if os.path.exists(save_path):
        print(f'Loading existing file at {save_path}')
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            df = data['processed_df']
            vocabulary = data['vocabulary']
    
    else :
        print(f'No file found, preprocessing...')
        df, vocabulary = preprocess_file(filename, save_dir)
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    
    # Get ground truth events
    ground_truth = (df.groupby(pd.Grouper(key='Timestamp', freq='1min'))['EventType']
                   .max()  # If any event in the minute window, count it as event
                   .fillna(0)
                   .astype(int))
    

    processed_df = df
    
    # Initialize results storage
    results = []
    previous_periods = []
    
    print("Analyzing time windows...")
    # Group by 1-minute windows
    for window_start, window_df in processed_df.groupby(
        pd.Grouper(key='Timestamp', freq='1min')
    ):  
        print(f"Analyzing time at {window_start}, window size {len(window_df)}")
        if len(window_df) == 0:
            print(f'No tweets at time {window_start}')
            results.append({
                'MatchID': df['MatchID'].iloc[0],
                'Timestamp': window_start,
                'is_event': 0,
                'score': 0,
                'true_event': ground_truth.get(window_start, 0)
            })
            continue
        # Process current window
        print(f'Processing time window')
        current_period = process_time_window(window_df, vocabulary)
        
        # Detect events
        print(f'Detecting event')
        is_event, score = detect_event(
            current_period,
            previous_periods,
            threshold=0.5
        )
        
        # Store results with ground truth
        results.append({
            'MatchID': df['MatchID'].iloc[0],
            'Timestamp': window_start,
            'is_event': int(is_event),
            'score': score,
            'true_event': ground_truth.get(window_start, 0)
        })
        
        # Update previous periods
        previous_periods.append(current_period)
        if len(previous_periods) > 5:
            previous_periods.pop(0)
    
    return pd.DataFrame(results)

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
def main() :
    """Process all match files and calculate metrics"""
    check_memory()
    data_dir = "challenge_data/train_tweets"
    all_true = []
    all_pred = []
    match_metrics = {}
    
    # Process each file
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        match_results = process_single_match_file(file_path)
        
        if match_results is not None:
            # Calculate metrics for this match
            metrics = calculate_metrics(
                match_results['true_event'], 
                match_results['is_event']
            )
            
            match_id = match_results['MatchID'].iloc[0]
            match_metrics[match_id] = metrics
            
            # Add to overall results
            all_true.extend(match_results['true_event'])
            all_pred.extend(match_results['is_event'])
            
            print(f"\nMetrics for match {match_id}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.3f}")
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(all_true, all_pred)
    
    print("\nOverall metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    return overall_metrics, match_metrics

if __name__ == "__main__":
    results = main()