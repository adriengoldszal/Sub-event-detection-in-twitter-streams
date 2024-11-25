# INF554
INF554 Machine and Deep Learning Data Challenge 

## TODO
Probablement ajouter une fonctionnalité de save la df après le preprocessing pour eviter d'attendre 22 min

## Installation instructions

Download challenge_data folder from the Kaggle and put it at the root.

Preprocessing : use a notebook to implement the different types of preprocessing you want to try and save the "period_features.csv" file in the work directory (example in preprocesser.ipynb).
Do not forget to also preprocess the whole evaluating data (template in preprocesser.ipynb) at once and save the "period_features_test.csv" file.
This is for us to be able to test separatelly the preprocessing and classifying parts of the project without haing to run everything (which takes time).

Classifying : import the "period_features.csv" training data and "period_features_test.csv" evaluating data and test the classifier you want. To export the data to be able to upload the correct format, look at classifier.ipynb.

