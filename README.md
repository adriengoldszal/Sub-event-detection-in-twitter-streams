# INF554 Team Petit Pois

INF554 Machine and Deep Learning Data Challenge : Sub-event detection in twitter streams. 

# Installation Instructions

1) Download and unzip the challenge_data folder from kaggle at the root, so it creates a `challenge_data` folder with the necessary train and test tweets, we should then have directories like this : `/challenge_data/train_tweets` and `challenge_data/eval_tweets`.


2) Create a virtual environment with the `requirements.txt` file. A somewhat recent version of python would be preferred >3.9 for everything to work smoothly down the line. 
    - run `python -m venv venv` in terminal at the source of the repository
    - then run `venv\Scripts\activate`
    - finally `pip install -r requirements.txt`

3) You're ready to run !

# Execution Intructions

## Machine-Learning methods

1) **Data preprocessing**, run `preprocesser_embedder.py`
This will output the period features for three different models (GloVe-50, GloVe-200, BERT) for both the training and evaluation data. These period features are located in `./preprocessed-data`. Preprocessing and embedding is a time-consuming task. We left our already preprocessed and embedded data in the folder to use directly for classification comparison.

2) **Classification** To compare the different classifiers, run `compare_classifier.py` that assumes the period features are located in "./preprocessed-data" and have the name given to them in the `preprocesser_embedder.py` file. This will output all the different predictions for the evaluation data in the folder "./predictions" and will also output in the work folder a file named "classifier_comparison.csv" which gives, for each model, accuracy results on a testing data selected at random.

## DSPy

### Creating the summaries 

This code has been taken from the great work by Polykarpos Meladianos, Christos Xypolopoulos, Giannis Nikolentzos, and Michalis Vazirgiannis in their paper [An Optimization Approach for Sub-event
Detection and Summarization in Twitter](https://www.lix.polytechnique.fr/~nikolentzos/files/meladianos_ecir18) to use their graph based summarization module. It has only been slightly adapted for testing purposes. 

1) Run the `optimization-sub-event-detection\filter.py` file to transform the train and evaluation data into data processable by the summarization module. 

2) For each match to be summarized, call in the following format `python optimization-sub-event-detection\main.py .\challenge_data\eval_tweets\converted_files\converted_GreeceIvoryCoast44.csv --output .\summaries --v`  
The summaries can all be found already in the `summaries` folder. 

3) Run `clean_dspy.py` which relies on the `formatting_summary.py` file located in the work folder as well as the already converted summaries (the `converted_MATCHNAME.txt` files) located in the `./summaries` folder. This will print on the terminal results for the trained module on the training data and testing data selected at random from the known data and it will also output in `./predictions` the predictions for the evaluation set.