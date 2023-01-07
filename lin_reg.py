""" This module uses Linear regression to predict student scores using the
 given parameters on students. In this part, the scores(g1, g2, g3) will be calculated as a function
 of the parameters given in the dataset.
 Refer to dataset for better understanding."""

# Import libraries for data processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Read dataset from csv file into a dataframe
df = pd.read_csv('student-mat.csv')

# Make column headings lower case and replaces spaces between strings with underscores
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Get all indexes of datatype object
strings = list(df.dtypes[df.dtypes == 'object'].index)

# Replace all spaces between strings in the dataset with underscores and make all entries lower case.
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Get length of dataset and use it to determine the length of training, validation and test datasets
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

# Shuffle indexes in the range of the dataset and set a random seed to make dataset more deterministic
idx = np.arange(n)
np.random.seed(2)
np.random.shuffle(idx)

# Split shuffled dataset into training, validation and test dataset
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

# Drop original indexes of the dataset from the dataframe
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Seperate the result from the parameters to be used for training 
y_train = df_train[['g1', 'g2', 'g3']].values
y_val = df_val[['g1', 'g2', 'g3']].values
y_test = df_test[['g1', 'g2', 'g3']].values

# Remove result columns from the original dataset
del df_train['g1']
del df_train['g2']
del df_train['g3']
del df_val['g1']
del df_val['g2']
del df_val['g3']
del df_test['g1']
del df_test['g2']
del df_test['g3']

# Get base parameters for training. This parameters are numerical columns
base = [ 'age', 'medu', 'fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime',
    'goout', 'dalc', 'walc', 'health', 'absences',]

# Get other categorical parameters for training
categorical_columns = [
    'school', 'sex', 'address', 'famsize', 'pstatus', 'mjob', 'fjob', 'reason', 'guardian', 
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet', 'romantic', ]

categorical = {}

# Get top 5 indexes for each categorical column into a dictionary
for c in categorical_columns:
    categorical[c] = list(df_train[c].value_counts().head().index)

# Prepare the dataset
def prepare_X(df):
    df = df.copy()

    features = []
    features = features + base

    for name, values in categorical.items():
        for value in values:
            df['%s_%s' % (name, value)] = (df[name] == value).astype(int)
            features.append('%s_%s' % (name, value))

    df_num = df[features]
    X = df_num.values
    return X

# Train the model using linear regression
def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

# Calculate the error 
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


# Tune the model to get the best value for r
# for r in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
#     X_train = prepare_X(df_train)
#     w0, w = train_linear_regression_reg(X_train, y_train, r=r)
#     X_val = prepare_X(df_val)
#     y_pred = w0 + X_val.dot(w)
#     score = rmse(y_val, y_pred)  
#     print(r, w0, score)

# Use the best value of r to train the model
# r = 0.01
# X_train = prepare_X(df_train)
# w0, w = train_linear_regression_reg(X_train, y_train, r=r)

# Prepare the validation dataset and use it to obtain the error on the best performing model
# X_val = prepare_X(df_val)
# y_pred = w0 + X_val.dot(w)
# score = rmse(y_val, y_pred)
# print(r, w0, score)

# Use both the test and validation dataset to train the model for better results
df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)
X_full_train = prepare_X(df_full_train)
y_full_train = np.concatenate([y_train, y_val])
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

# Prepare the test dataset and calculate the error
X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)
print(score)

# Testing the models predictions using an entry from the test dataset
student = df_test.iloc[20].to_dict()
df_small = pd.DataFrame([student])
X_small = prepare_X(df_small)
y_pred = w0 + X_small.dot(w)
# y_pred = y_pred[0]
print(y_pred)
print(y_test[20])

