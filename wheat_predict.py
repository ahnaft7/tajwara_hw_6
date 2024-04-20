"""
Ahnaf Tajwar
Class: CS 677
Date: 4/20/24
Homework Problems # 1-3
Description of Problem (just a 1-2 line summary!): These problems are to compare the different SVM models with a classifier of my choice (Logistic Regression) as well as with the k-means clustering model.
    It also goes through checking the accuracy of the k-means clustering by calculating the euclidean distance with the centroids and data.
"""
from collections import Counter
import random
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
    y_means = kmeans.fit_predict(X_train)
    distortions.append(sum(np.min(cdist(X_train, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])
    inertias.append(kmeans.inertia_)

# Plot the distortions
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Distortion vs k')
# plt.show()
plt.savefig('distortion_plot.png')

# Best k using the "knee" method
best_k = 3
print(f"\nBest k using the knee method: {best_k}")

print("\n-------------Random fi and fj with k = 3--------------")

# features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
# # Pick a random item from the list
# fi = random.choice(features)
# # Remove the picked item from the list
# features.remove(fi)
# fj = random.choice(features)
fi = 'f5'
fj = 'f4'
print("fi: ", fi)
print("fj: ", fj)
feature_list = [fi, fj]

# Test Train split for whole dataset
X = df[feature_list]
Y = df[['L']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
Y_train = np.ravel(Y_train)
Y_test = np.ravel(Y_test)
print("training data: ", Y_train)

kmeans = KMeans(n_clusters=best_k, init='random', random_state=1)
y_means = kmeans.fit_predict(X_train)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("\nlabels: ", labels)

plt.clf()

# Plot labels
for label in np.unique(labels):
    plt.scatter(X_train.loc[labels == label, fi], X_train.loc[labels == label, fj], label=f'Cluster {label + 1}')

# Plot centroids with different colors
centroid_colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
for i, centroid in enumerate(centroids):
    plt.scatter(centroid[X_train.columns.get_loc(fi)], centroid[X_train.columns.get_loc(fj)], marker='x', s=200, c=centroid_colors[i], label=f'Centroid {i+1}')

plt.xlabel(fi)
plt.ylabel(fj)
plt.title(f'Data Points and Centroids ({fi} vs {fj})')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('fifj_class_centroids_plot.png')

# Initialize a dictionary to store the majority class for each cluster
cluster_to_class = {}

# Iterate over each cluster
for cluster_label in np.unique(y_means):
    # Get the true class labels within the current cluster
    cluster_points = Y_train[y_means == cluster_label]
    print("\nAll cluster points: ", cluster_points)
    # Count the occurrences of each original class label
    class_counts = Counter(cluster_points)
    print(f"\nnumber of each class label for cluster label {cluster_label}: ", class_counts)
    
    # Find the majority class label within the current cluster
    majority_class = class_counts.most_common(1)[0][0]
    print("\nmajority: ", majority_class)

    cluster_to_class[cluster_label] = majority_class

# Print centroids and assigned labels for each cluster
for i, centroid in enumerate(centroids):
    majority_class = cluster_to_class[i]
    print(f"\nCluster {i+1}:")
    print("Centroid:", centroid)
    print("Assigned Label:", majority_class)
    print()

print("-------------Overall Accuracy of New Classifier Based on Distance to Centroid--------------")
# Initialize variables to count correct predictions and total predictions
correct_predictions = 0
total_predictions = len(X_test)
X_test.reset_index(drop=True, inplace=True)
# Iterate through each data point
for i, x in X_test.iterrows():
    # Calculate the Euclidean distance from the data point to each centroid
    distances_A = np.linalg.norm(x - centroids[0])
    distances_B = np.linalg.norm(x - centroids[1])
    distances_C = np.linalg.norm(x - centroids[2])
    
    # Find the nearest centroid
    nearest_centroid = np.argmin([distances_A, distances_B, distances_C])
    
    # Assign label based on the nearest centroid
    if nearest_centroid == 0:
        predicted_label = 1
    elif nearest_centroid == 1:
        predicted_label = 2
    else:
        predicted_label = 3
    
    # Compare predicted label with true label
    true_label = Y_test[i]
    if predicted_label == true_label:
        correct_predictions += 1

# Calculate overall accuracy
accuracy = correct_predictions / total_predictions

print("Overall Accuracy:", accuracy)

# New Classifer compared to Original k-means
print("\n-------------Compare New Classifier with Original k-means--------------")

X = filtered_df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']]
Y = filtered_df[['L']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
Y_train = np.ravel(Y_train)
Y_test = np.ravel(Y_test)

# Calculate distortions for k = 1 to 8
distortions = []
inertias = []
K = range(1, 9)
for k in K:
    kmeans = KMeans(n_clusters=k, init='random', random_state=1)
    y_means = kmeans.fit_predict(X_train)
    distortions.append(sum(np.min(cdist(X_train, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])
    inertias.append(kmeans.inertia_)

# Plot the distortions
plt.clf()
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Distortion vs k')
# plt.show()
plt.savefig('distortion_plot_2_labels.png')

# Best k using the "knee" method
best_k = 2
print(f"\nBest k using the knee method: {best_k}")
print("training data: ", Y_train)

kmeans = KMeans(n_clusters=best_k, init='random', random_state=1)
y_means = kmeans.fit_predict(X_train)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("\nlabels: ", labels)

# Initialize a dictionary to store the majority class for each cluster
cluster_to_class = {}

# Iterate over each cluster
for cluster_label in np.unique(y_means):
    # Get the true class labels within the current cluster
    cluster_points = Y_train[y_means == cluster_label]
    print("\nAll cluster points: ", cluster_points)
    # Count the occurrences of each original class label
    class_counts = Counter(cluster_points)
    print(f"\nnumber of each class label for cluster label {cluster_label}: ", class_counts)
    
    # Find the majority class label within the current cluster
    majority_class = class_counts.most_common(1)[0][0]
    print("\nmajority: ", majority_class)

    cluster_to_class[cluster_label] = majority_class

# Print centroids and assigned labels for each cluster
for i, centroid in enumerate(centroids):
    majority_class = cluster_to_class[i]
    print(f"\nCluster {i+1}:")
    print("Centroid:", centroid)
    print("Assigned Label:", majority_class)
    print()

# Initialize variables to count correct predictions and total predictions
correct_predictions = 0
total_predictions = len(X_test)
X_test.reset_index(drop=True, inplace=True)
true_labels = []
predicted_labels = []
# Iterate through each data point
for i, x in X_test.iterrows():
    # Calculate the Euclidean distance from the data point to each centroid
    distances_A = np.linalg.norm(x - centroids[0])
    distances_B = np.linalg.norm(x - centroids[1])
    
    # Find the nearest centroid
    nearest_centroid = np.argmin([distances_A, distances_B])
    
    # Assign label based on the nearest centroid
    if nearest_centroid == 0:
        predicted_label = 1
    elif nearest_centroid == 1:
        predicted_label = 2
    
    # Compare predicted label with true label
    true_label = Y_test[i]
    if predicted_label == true_label:
        correct_predictions += 1
    
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)


# Calculate overall accuracy
accuracy = correct_predictions / total_predictions

print("Overall Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

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