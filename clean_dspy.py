import dspy
import numpy as np
import pandas as pd
import openai
from tqdm import tqdm
from dspy.evaluate.metrics import answer_exact_match
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from formatting_summary import *
import time

### Setting up DSPy
openai.api_key = "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"
lm=dspy.OpenAI(model="gpt-4o-mini")
dspy.settings.configure(lm=lm)
print(lm("Hello, world!"))
### Defining the tasks I want the program to do
class DetectEvent(dspy.Signature):
    """Detect whether an event like 'full time', 'end of match', 'goal', 'half time', 'kick off', 'other', 'owngoal', 'penalty', 'red card', 'yellow card' happened."""
    tweet = dspy.InputField(desc="the tweet to analyze")
    answer = dspy.OutputField(desc="whether an event occurred. Should be a boolean (1 or 0).")

class DetectEventImpl(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought(DetectEvent)
    def forward(self, tweet):
        return self.predict(tweet=tweet)
    
predict = DetectEventImpl()

available_matches = ["ArgentinaGermanyFinal77","AustraliaNetherlands29","AustraliaSpain34","BelgiumSouthKorea59","CameroonBrazil36","FranceGermany70","FranceNigeria66","GermanyAlgeria67","GermanyBrazil74","GermanyUSA57","HondurasSwitzerland54","MexicoCroatia37","NetherlandsChile35","PortugalGhana58","USASlovenia2010"]
n_total_matches = len(available_matches)
n_training_matches = int(n_total_matches*0.34)
n_testing_matches = 3
np.random.seed(42)
available_matches = np.random.permutation(available_matches)
print(f"Number of total matches: {n_total_matches}")
print(f"Number of matches used in training (or rather optimizing): {n_training_matches}")
print(f"Number of matches used in testing: {n_testing_matches}")

### Training/optimizing the program
training_matches = available_matches[:n_training_matches] # matches for training
train_dfs = []
for match in training_matches:
    train_df = get_correct_summary_df(match,train_or_eval='train')
    train_dfs.append(train_df)
train_data = pd.concat(train_dfs)

trainset = [dspy.Example(tweet=x["Content"],answer=str(x["EventType"])).with_inputs("tweet") for x in train_data.to_dict(orient="records")]
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=4)
teleprompter2 = BootstrapFewShotWithRandomSearch(metric=answer_exact_match, **config)
print("Training the program...")
beginning = time.time()
optimized_program = teleprompter2.compile(predict, trainset=trainset)
print(f"Training took {time.time()-beginning} seconds.")


### Evaluating the program on the training data
print("Evaluating the program on the training data...")
beginning = time.time()
tqdm.pandas()
train_data["EventTypePredicted"] = train_data["Content"].progress_apply(
        lambda x: int(optimized_program(tweet=x).answer)
    )
print(f"Testing took {time.time()-beginning} seconds.")
confusion_matrix = pd.crosstab(train_data['EventType'], train_data['EventTypePredicted'], rownames=['EventType'], colnames=['EventTypePredicted'], dropna=False)

print(f"Confusion matrix for the training data (i.e. matches {training_matches}):")
print(confusion_matrix)
print(f"Accuracy: {np.diag(confusion_matrix).sum()/confusion_matrix.sum().sum()}")

### Predicting on the testing data
testing_match = available_matches[n_training_matches:n_training_matches+n_testing_matches] # matches for testing
test_dfs = []
for match in testing_match:
    test_df = get_correct_summary_df(match,train_or_eval='train')
    test_dfs.append(test_df)
test_data = pd.concat(test_dfs)
print("Predicting on the testing data...")
beginning = time.time()
tqdm.pandas()
test_data["EventTypePredicted"] = test_data["Content"].progress_apply(
        lambda x: int(optimized_program(tweet=x).answer)
    )
print(f"Prediction took {time.time()-beginning} seconds.")
confusion_matrix_test = pd.crosstab(test_data['EventType'], test_data['EventTypePredicted'], rownames=['EventType'], colnames=['EventTypePredicted'], dropna=False)
print(f"Confusion matrix for the testing data (i.e. match {testing_match}):")
print(confusion_matrix_test)
print(f"Accuracy: {np.diag(confusion_matrix_test).sum()/confusion_matrix_test.sum().sum()}")

### Predicting on the evaluation set
eval_matches = ["GermanyGhana32", "GermanySerbia2010", "NetherlandsMexico64", "GreeceIvoryCoast44"]
print("Predicting on the evaluation set...")
beginning = time.time()
eval_dfs = []
for match in eval_matches:
    eval_df = get_correct_summary_df(match,train_or_eval='eval')
    tqdm.pandas()
    # Apply the model with a progress bar
    eval_df["EventTypePredicted"] = eval_df["Content"].progress_apply(
        lambda x: int(optimized_program(tweet=x).answer)
    )
    eval_dfs.append(eval_df)
    
eval_df = pd.concat(eval_dfs)
eval_df = eval_df[["ID","EventTypePredicted"]]
print(f"Prediction took {time.time()-beginning} seconds.")
eval_df.to_csv("predictions/eval_dspy_summarised_trained_predictions.csv", index=False)

print("Done!")