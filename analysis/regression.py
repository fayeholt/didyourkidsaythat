import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'/Users/madelineholt/didyourkidsaythat/data/encoded_dataset.csv')
print(df.head())

# defining the dependent and independent variables
df = df.dropna(how='any')
X = df[['AC', 'freq', 'age']]
y = df[['upvote_percent']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# X_train = X
# y_train = y

log_reg = sm.OLS(y_train, X_train).fit()

# printing the summary table
print(log_reg.summary())

# predictions on test data
# performing predictions on the test dataset
yhat = log_reg.predict(X_test)
prediction = list(map(round, yhat))

# comparing original and predicted values of y
print('Actual values', list(y_test.values))
print(list(X_test['age']))
print('Predictions :', prediction)

# confusion matrix
cm = confusion_matrix(y_test, prediction)
print("Confusion Matrix : \n", cm)

# accuracy score of the model
print('Test accuracy = ', balanced_accuracy_score(y_test, prediction))

print('F1 = ', f1_score(y_test, prediction))