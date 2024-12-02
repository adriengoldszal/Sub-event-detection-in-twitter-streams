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
import psutil
import resource
from scipy.sparse import csr_matrix

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def print_memory_info():
    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / (1024.0 ** 3):.1f} GB")
    print(f"Available memory: {memory.available / (1024.0 ** 3):.1f} GB")
    print(f"Used memory: {memory.used / (1024.0 ** 3):.1f} GB")
    print(f"Memory percentage used: {memory.percent}%")
    
def check_memory():
    print("\nMemory Information:")
    print_memory_info()
    process = psutil.Process()
    print(f"Current process memory: {process.memory_info().rss / (1024.0 ** 3):.2f} GB")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(f"Memory limits - soft: {soft/(1024**3):.1f} GB, hard: {hard/(1024**3):.1f} GB\n")
    
    
def generate_adjacency_matrix_dense(tweets, vocabulary):
    print(f'Generating adjacency matrix')
    print_memory_usage()
    """Method that is used to generate the adjacency matrix of the given tweets"""
    wordsNumber = len(vocabulary)
    adjacency_matrix = np.zeros((wordsNumber, wordsNumber))
    tweets_edges = []
    
    for _, tweet in tweets['Tweet'].items():
        # Convert tweet tokens to set to remove duplicates
        tweet_words = set(tweet.split())  # Assuming tweet is space-separated string
        
        # Get indices of words in vocabulary
        indexes = [bisect.bisect_left(vocabulary, word) for word in tweet_words]
        
        # Initialize edges list for this tweet
        tweet_edges = []
        
        # Create edges between all word pairs
        for i, idx1 in enumerate(indexes):
            for j, idx2 in enumerate(indexes[i+1:], i+1):  # Start from i+1 to avoid duplicates
                if idx1 != idx2:
                    # Add weight to adjacency matrix
                    weight = 1.0 / len(tweet_words)
                    adjacency_matrix[idx1, idx2] += weight
                    adjacency_matrix[idx2, idx1] += weight
                    
                    # Add edge to tweet edges
                    tweet_edges.append(sorted([vocabulary[idx1], vocabulary[idx2]]))
        
        tweets_edges.append(tweet_edges)
    
    return adjacency_matrix, tweets_edges

def get_edges_weight(adjacency_matrix, vocabulary, edges_list, nodes_list):
    """Method that is used to extract the weight for each edge in the given list. The nodes_list parameter is a
    list that contains the nodes that are included in the given edges """
    
    print(f'Getting edge weights ')
    print_memory_usage()
    nodes = {}
    for node in nodes_list:
        index = bisect.bisect(vocabulary, node) - 1
        if (0 <= index <= len(vocabulary)) and vocabulary[index] == node:
            nodes[node] = index

    weight_list = []
    for edge in edges_list:
        first_word, second_word = edge[0], edge[1]
        if all(word in nodes for word in (first_word, second_word)):
            indexes = [nodes[first_word], nodes[second_word]]
            indexes.sort()
            weight_list.append(adjacency_matrix[indexes[0], indexes[1]])
        else:
            weight_list.append(0)
    return weight_list


def get_nonzero_edges(matrix):
    """Method that is used to extract from the adjacency matrix the edges with no-negative weights"""
    print(f'Getting non zero edges')
    print_memory_usage()
    rows, columns, values = find(matrix)
    return [[rows[i], columns[i], float(values[i])] for i in range(len(rows))]

def generate_vector(adjacency_matrix, vocabulary):
    """Method that is used to generate a vector for the current period"""
    print(f'Generating vector')
    print_memory_usage()
    
    non_zero_edges = get_nonzero_edges(adjacency_matrix)
    vector = np.zeros((len(non_zero_edges), 1))
    vector_edges = []
    vector_nodes = set()
    weighted_edges = {}
    counter = 0
    for row, column, value in non_zero_edges:
        vector[counter] = value
        nodes = [vocabulary[row], vocabulary[column]]
        vector_edges.append(nodes)
        vector_nodes.update(nodes)
        weighted_edges[tuple(sorted(nodes))] = value
        counter += 1
    return vector, vector_nodes, vector_edges, weighted_edges


def detect_event(current_period_data, previous_periods_data, threshold=0.5):
    """
    Detect if current period contains an event using Least Squares Optimization.
    
    Args:
        current_period_data: Dict containing current period's data 
            (adjacency_matrix, vector, vector_nodes, vector_edges, tweets_edges, etc.)
        previous_periods_data: List of dicts containing previous periods' data
        threshold: Event detection threshold
    
    Returns:
        Tuple of (is_event, period_score, summary)
    """
    if len(current_period_data['tweets_edges']) == 0:
        return False, -1, "No tweets found in the current period."
        
    period_score = -1
    if previous_periods_data:
        # Get weights matrix comparing current period to previous periods
        weights = np.zeros((len(current_period_data['vector_edges']), 
                          len(previous_periods_data)))
        
        for i, prev_period in enumerate(previous_periods_data):
            weights[:, i] = np.asarray(
                get_edges_weight(
                    prev_period['adjacency_matrix'],
                    prev_period['vocabulary'],
                    current_period_data['vector_edges'],
                    current_period_data['vector_nodes']
                )
            )
        
        # Optimize to get period score
        print(f'Optimizing least squares')
        period_score = optimize_least_squares(weights, current_period_data['vector'])
    
    is_event = period_score >= threshold
    
    return is_event, period_score

def optimize_least_squares(A, b):
    """Match paper's implementation exactly"""
    print(f'Optimizing least squares')
    print_memory_usage()
    
    # Convert A to sparse format if large and sparse
    A = csr_matrix(A)  # This makes A sparse
    
    x = Variable(A.shape[1])
    print(f"Minimizing norm")
    
    # Use the @ operator for matrix multiplication
    objective = Minimize(norm(A @ x - b))
    print_memory_usage()
    
    print(f"Constraints")
    constraints = [0 <= x, sum(x) == 1]
    print_memory_usage()
    
    print(f'Problem')
    problem = Problem(objective, constraints)
    minimum = problem.solve()
    print_memory_usage()
    
    print(f'Calculate value')
    value = A.dot(x.value) - b
    value[value > 0] = 0
    minimum = np.linalg.norm(value)
    return minimum
