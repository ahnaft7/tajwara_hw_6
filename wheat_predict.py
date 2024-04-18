"""
Ahnaf Tajwar
Class: CS 677
Date: 4/20/24
Homework Problems # 1-3
Description of Problem (just a 1-2 line summary!): These problems are to 
"""
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Define column names
column_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'L']

# Read the file using a regular expression pattern as the delimiter
df = pd.read_csv("seeds_dataset.txt", delimiter='\s+', header=None, names=column_names)

print(df)

buid = 0
R = buid % 3
print(f'\nR = {R}: class L = 1 (negative) and L = 2 (positive)')

# Filter rows where L is 1 or 2
filtered_df = df[df['L'].isin([1, 2])]

# Print the filtered DataFrame
print(filtered_df)

# Test Train split for filtered_df
X = filtered_df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']]
Y = filtered_df[['L']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
Y_train = np.ravel(Y_train)
Y_test = np.ravel(Y_test)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Dictionary for metrics
model_metrics = {}

#---------------------linear kernel SVM-------------------

print("\n-------------linear kernel SVM--------------")

# Initialize the SVM classifier with linear kernel
svm_classifier = SVC(kernel='linear')

# Train the classifier on the training data
svm_classifier.fit(X_train, Y_train)

# Predict labels for the test set
Y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred, labels=np.unique(Y))

print("Confusion Matrix: (1 is Negative, 2 is Positive)")
print("\t\tPredicted labels")
print("\t\t 1    2")
print("Actual labels 1", conf_matrix[0])
print("              2", conf_matrix[1])

tp = conf_matrix[1][1]
fn = conf_matrix[1][0]
fp = conf_matrix[0][1]
tn = conf_matrix[0][0]
print("TP: ", tp)
print("FP: ", fp)
print("TN: ", tn)
print("FN: ", fn)

# Calculate True Positive Rate
tpr = tp / (tp + fn)

# Calculate True Negative Rate
tnr = tn / (tn + fp)

print("True Positive Rate:", tpr)
print("True Negative Rate:", tnr)

model_metrics['linear SVM'] = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'accuracy': accuracy, 'TPR': tpr, 'TNR': tnr}

#---------------------Gaussian kernel SVM-------------------

print("\n-------------Gaussian kernel SVM--------------")

# Initialize the SVM classifier with Gaussian (RBF) kernel
svm_classifier = SVC(kernel='rbf')

# Train the classifier on the training data
svm_classifier.fit(X_train, Y_train)

# Predict labels for the test set
Y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix: (1 is Negative, 2 is Positive)")
print("\t\tPredicted labels")
print("\t\t 1    2")
print("Actual labels 1", conf_matrix[0])
print("              2", conf_matrix[1])

tp = conf_matrix[1][1]
fn = conf_matrix[1][0]
fp = conf_matrix[0][1]
tn = conf_matrix[0][0]
print("TP: ", tp)
print("FP: ", fp)
print("TN: ", tn)
print("FN: ", fn)

# Calculate True Positive Rate
tpr = tp / (tp + fn)

# Calculate True Negative Rate
tnr = tn / (tn + fp)

print("True Positive Rate:", tpr)
print("True Negative Rate:", tnr)

model_metrics['Gaussian SVM'] = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'accuracy': accuracy, 'TPR': tpr, 'TNR': tnr}

#--------------------- polynomial kernel SVM of degree 3-------------------

print("\n------------- polynomial kernel SVM of degree 3--------------")

# Initialize the SVM classifier with polynomial kernel of degree 3
svm_classifier = SVC(kernel='poly', degree=3)

# Train the classifier on the training data
svm_classifier.fit(X_train, Y_train)

# Predict labels for the test set
Y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix: (1 is Negative, 2 is Positive)")
print("\t\tPredicted labels")
print("\t\t 1    2")
print("Actual labels 1", conf_matrix[0])
print("              2", conf_matrix[1])

tp = conf_matrix[1][1]
fn = conf_matrix[1][0]
fp = conf_matrix[0][1]
tn = conf_matrix[0][0]
print("TP: ", tp)
print("FP: ", fp)
print("TN: ", tn)
print("FN: ", fn)

# Calculate True Positive Rate
tpr = tp / (tp + fn)

# Calculate True Negative Rate
tnr = tn / (tn + fp)

print("True Positive Rate:", tpr)
print("True Negative Rate:", tnr)

model_metrics['polynomial SVM'] = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'accuracy': accuracy, 'TPR': tpr, 'TNR': tnr}

#---------------------Logistic Regression-------------------

print("\n-------------Logistic Regression--------------")

# Test Train split for filtered_df
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
class_labels = [1, 2]

X = filtered_df[features].values

le = LabelEncoder()
Y = le.fit_transform(filtered_df['L'].values)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
# Y_train = np.ravel(Y_train)
# Y_test = np.ravel(Y_test)

# Initialize the logistic regression model
logistic_regression = LogisticRegression()

# Train the model on the training data
logistic_regression.fit(X_train, Y_train)

# Predict labels for the test set
y_pred = logistic_regression.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(Y_test, y_pred)

print("Confusion Matrix: (1 is Negative, 2 is Positive)")
print("\t\tPredicted labels")
print("\t\t 1    2")
print("Actual labels 1", conf_matrix[0])
print("              2", conf_matrix[1])

tp = conf_matrix[1][1]
fn = conf_matrix[1][0]
fp = conf_matrix[0][1]
tn = conf_matrix[0][0]
print("TP: ", tp)
print("FP: ", fp)
print("TN: ", tn)
print("FN: ", fn)

# Calculate True Positive Rate
tpr = tp / (tp + fn)

# Calculate True Negative Rate
tnr = tn / (tn + fp)

print("True Positive Rate:", tpr)
print("True Negative Rate:", tnr)

model_metrics['Logistic Regression'] = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'accuracy': accuracy, 'TPR': tpr, 'TNR': tnr}

# Create table of metrics
metrics_df = pd.DataFrame(model_metrics).T

print("\n", metrics_df)

#---------------------k-means Clustering-------------------

print("\n-------------k-means Clustering--------------")

# Test Train split for whole dataset
X = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']]
Y = df[['L']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
Y_train = np.ravel(Y_train)
Y_test = np.ravel(Y_test)

# Calculate distortions for k = 1 to 8
distortions = []
inertias = []
K = range(1, 9)
for k in K:
    kmeans = KMeans(n_clusters=k, init='random', random_state=1)
    kmeans.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeans.inertia_)

# Plot the distortions
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Distortion vs k')
plt.show()

# Best k using the "knee" method
print("Best k using the knee method: 3")

plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Inertia vs k')
plt.show()


