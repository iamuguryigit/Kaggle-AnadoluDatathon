import pandas as pd


train_path = "/../train-utf8.csv"
test_path = "/../test-utf8 (1).csv"
submission_path = "/../samplesubmission.csv"



train = pd.read_csv(train_path, low_memory=False)
test = pd.read_csv(test_path, low_memory=False)
submission = pd.read_csv(submission_path, low_memory=False)






