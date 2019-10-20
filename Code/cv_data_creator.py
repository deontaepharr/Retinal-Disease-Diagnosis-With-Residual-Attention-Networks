import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import glob
from pathlib import Path

# Construct the files to be placed inside a dataframe
def construct_data(filepath):
    data = {}
    data['filepath'] = filepath
    data['class'] = str(Path(filepath).parent).split('/')[-1]
    return data

data_filepath = "/pylon5/cc5614p/deopha32/eye_images/"
training_filepath = "/home/deopha32/EyeDiseaseClassification/Data/training_data_{}"
validation_filepath = "/home/deopha32/EyeDiseaseClassification/Data/validatation_data_{}"

full_training_filepath = "/home/deopha32/EyeDiseaseClassification/Data/training_data"
testing_filepath = "/home/deopha32/EyeDiseaseClassification/Data/testing_data"

data_files = [filename for filename in glob.iglob(data_filepath + '*/*/*', recursive=True)]

# Place inside dataframe
data = [construct_data(file) for file in data_files]
model_data = pd.DataFrame(data)

X = model_data['filepath']
y = model_data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

count = 1
skf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model_train_data = pd.DataFrame(pd.concat([X_train_fold, y_train_fold], axis=1))
    model_val_data = pd.DataFrame(pd.concat([X_val_fold, y_val_fold], axis=1))

    model_train_data.to_csv(training_filepath.format(count), index=False)
    model_val_data.to_csv(validation_filepath.format(count), index=False)
    
    count += 1

model_full_train_data = pd.DataFrame(pd.concat([X_train, y_train], axis=1))
model_full_train_data.to_csv(full_training_filepath, index=False)

model_test_data = pd.DataFrame(pd.concat([X_test, y_test], axis=1))
model_test_data.to_csv(testing_filepath, index=False)