# DataMining_EX_serial1
---
# 1. description for EX1 : 

# Code Breakdown

### 1. Imports


### The code starts by importing the necessary libraries:

    pandas: For data manipulation, especially for reading and processing the dataset.

    numpy: For numerical operations such as matrix manipulation, distance calculations, and more.

    xlrd: For reading Excel files (although pandas also handles 
    this through read_excel).

    math: For mathematical operations like logarithms.

    collections.Counter: For counting frequencies in lists (used to calculate entropy).


### 2. Loading the Dataset

The code reads an Excel file located at file_path using two methods:

    pandas.read_excel() loads the data into a DataFrame, which 

    is then converted into a numpy array (data_array).

    xlrd.open_workbook() loads the workbook and iterates through 

    the rows to create a list of lists (data1).


### 3. Entropy Calculation

Entropy is a measure of uncertainty or disorder, often used in information theory.

    entropy() function: Calculates the entropy of a feature 

    (column) in the dataset. It uses the formula:

    H(X)=−∑P(xi)log⁡2P(xi)

    H(X)=−∑P(xi​)log2​P(xi​)


    where P(xi)P(xi​) is the probability of each unique value in 

    the feature. It uses Counter to get the frequencies and 

    calculates entropy based on these frequencies.

    calculate_entropy_for_features() function: Loops through 

    each feature (column) of the dataset and calculates the 

    entropy for each one, storing the results in a list.


### 4. Euclidean Distance

The Euclidean distance is used to measure the "straight-line" 

distance between two points in a multi-dimensional space.

    euclidean_distance_matrix() function: Takes the data as 

    input, and calculates the pairwise Euclidean distances 

    between all data points. It uses the formula:

    di,j=∑k=1n(xi,k−xj,k)2

    di,j​=k=1∑n​(xi,k​−xj,k​)2


    ​ where ii and jj are data points, and nn is the number of 

    features.

### 5. Covariance Calculation

Covariance measures how two variables change together. It is 

used in calculating the Mahalanobis distance and the correlation 

matrix.

    covariance() function: Manually calculates the covariance 

    between two variables XX and YY.

    covariance_matrix() function: Calculates the covariance 

    matrix for the entire dataset by calling covariance() for 

    each pair of features.

### 6. Matrix Inversion

To compute the Mahalanobis distance, we need to invert the 

covariance matrix.

    inverse_matrix() function: Uses Gaussian elimination to 

    calculate the inverse of a given matrix. It forms an 

    augmented matrix, applies row operations to convert it into 

    row echelon form, and then extracts the inverse from the 

    augmented matrix.


### 7. Mahalanobis Distance

The Mahalanobis distance is a measure of the distance between 

two points, taking into account the covariance structure of the 

dataset.

    mahalanobis_distance() function: Calculates the Mahalanobis 

    distance between two vectors, xx and yy, using the inverse 

    of the covariance matrix.

    mahalanobis_distance_matrix() function: Calculates the 

    pairwise Mahalanobis distances between all data points by 

    using the covariance matrix and its inverse.


### 8. Correlation Matrix

The correlation matrix measures the strength and direction of 

the linear relationship between pairs of features.

    correlation_matrix() function: Calculates the correlation 

    matrix for the dataset by normalizing the covariance matrix 

    with the standard deviations of the features.


### 9. Output

The script then generates and prints:

    The Euclidean distance matrix.

    The Mahalanobis distance matrix.

    The correlation matrix.

    The entropy for each feature.


Each of these matrices is printed for inspection.


# Data Analysis with Entropy, Distance Metrics, and Covariance

This project involves calculating various statistical measures 
and distance metrics on a dataset loaded from an Excel file. The following calculations are performed:
- **Entropy** for each feature (column) in the dataset.
- **Euclidean distance matrix** for pairwise distance between data points.
- **Mahalanobis distance matrix** considering the covariance of the dataset.
- **Correlation matrix** for each pair of features.

## Dependencies

The following Python libraries are required to run the script:
- `pandas` (for data manipulation)
- `numpy` (for numerical operations)
- `xlrd` (for reading Excel files)
- `math` (for mathematical operations)
- `collections.Counter` (for frequency counting)

You can install the dependencies using the following:
```bash
pip install pandas numpy xlrd
```

### How to Use

### 1. Prepare your Dataset

Ensure that your dataset is in an Excel file (.xls or .xlsx format). The script assumes that the data is in the first sheet of the workbook. The path to the dataset file must be specified in the file_path variable in the code.

```bash
file_path = r'path\to\your\dataset.xls'
```

### 2. Run the Script

To run the script, simply execute it in your Python environment. The script will load the data from the specified file, process it, and print the results:

    Euclidean Distance Matrix: Pairwise distances between data points.
    Mahalanobis Distance Matrix: Pairwise Mahalanobis distances, taking into account covariance.
    Correlation Matrix: Pairwise correlations between features.
    Entropy: Entropy value for each feature in the dataset.

### 3. Understanding the Output

    Euclidean Distance Matrix: A symmetric matrix where each element represents the Euclidean distance between two data points.
    Mahalanobis Distance Matrix: A symmetric matrix similar to the Euclidean distance matrix, but it takes into account the correlations between features.
    Correlation Matrix: A matrix that shows the linear relationship between features. Values close to 1 or -1 indicate strong relationships, while values near 0 indicate weak or no linear relationship.
    Entropy: A measure of uncertainty or disorder in each feature. Features with higher entropy have more variability.

## Functions Overview

### Entropy Functions


    entropy(values): Calculates the entropy of a single feature.
    calculate_entropy_for_features(data): Calculates entropy for each feature in the dataset.

### Distance Metric Functions

    euclidean_distance_matrix(data): Calculates the pairwise Euclidean distances between data points.
    mahalanobis_distance_matrix(data): Calculates the pairwise Mahalanobis distances between data points.

### Statistical Functions

    covariance(X, Y): Calculates the covariance between two variables.
    covariance_matrix(data): Calculates the covariance matrix for the dataset.
    correlation_matrix(data): Calculates the correlation matrix for the dataset.

### Matrix Functions

    inverse_matrix(matrix): Calculates the inverse of a matrix using Gaussian elimination.

### Example Output

```bash

Euclidean Distance Matrix:
[[0.         1.41421356 2.82842712]
 [1.41421356 0.         1.41421356]
 [2.82842712 1.41421356 0.        ]]

Mahalanobis Distance Matrix:
[[0.         1.41421356 2.82842712]
 [1.41421356 0.         1.41421356]
 [2.82842712 1.41421356 0.        ]]

Correlation Matrix:
[[1.         0.5        0.3       ]
 [0.5        1.         0.7       ]
 [0.3        0.7        1.        ]]

Entropy is:
Entropy of feature 1: 1.0
Entropy of feature 2: 0.8
```

## License

This project is licensed under the MIT License.


---

### Summary
This code is designed to perform a variety of statistical and distance-based analyses on a dataset. It includes entropy, covariance, distance calculations (Euclidean and Mahalanobis), and correlation matrices, and it prints out the results for further analysis. The `README.md` file explains how to set up the environment, use the script, and understand its output.

---

# @Author : babak yousefian


---


# 2. Description for EX2 : 
---

## part one is : 

# Breakdown of Code:


# Imports and Libraries

```bash

import openpyxl
```
openpyxl: This library is used to read Excel files. It is imported to load and process data from an Excel file.


### Gini Impurity Function

```bash

def gini_impurity(y):
    total = len(y)
    class_counts = {}
    for label in y:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    impurity = 1.0
    for count in class_counts.values():
        prob = count / total
        impurity -= prob ** 2
    return impurity
```

gini_impurity(y): Calculates the Gini Impurity for a given set of target labels (y).
Gini Impurity is a measure of how often a randomly chosen element from the dataset would be incorrectly classified.
The function counts the frequency of each label, then computes the Gini Impurity using the formula:
Gini=1−∑(pi2)
Gini=1−∑(pi2​) where pipi​ is the probability of each class label.


### Best Split Function

```bash

def best_split(X, y):
    best_gini = float('inf')
    best_split_feature = None
    best_split_value = None

    for feature_idx in range(len(X[0])):
        unique_values = sorted(set(row[feature_idx] for row in X))

        for value in unique_values:
            left_y = [y[i] for i in range(len(X)) if X[i][feature_idx] <= value]
            right_y = [y[i] for i in range(len(X)) if X[i][feature_idx] > value]

            left_gini = gini_impurity(left_y)
            right_gini = gini_impurity(right_y)
            gini = (len(left_y) * left_gini + len(right_y) * right_gini) / len(y)

            if gini < best_gini:
                best_gini = gini
                best_split_feature = feature_idx
                best_split_value = value

    return best_split_feature, best_split_value
```

best_split(X, y): Finds the best feature and value to split the dataset at based on the Gini Impurity criterion.
It iterates over each feature and finds the best value to partition the data into two groups that minimize the Gini Impurity.
The split is evaluated by calculating the Gini Impurity for the left and right subsets and combining them by their relative sizes.

### Build Tree (Hunt's Algorithm)

```bash

def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    if len(set(y)) == 1 or len(X) < min_samples_split or (max_depth is not None and depth >= max_depth):
        return {'class': max(set(y), key=y.count)}

    feature_idx, value = best_split(X, y)

    if feature_idx is None:
        return {'class': max(set(y), key=y.count)}

    left_X, left_y, right_X, right_y = [], [], [], []
    for i in range(len(X)):
        if X[i][feature_idx] <= value:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])

    left_tree = build_tree(left_X, left_y, depth + 1, max_depth, min_samples_split, min_samples_leaf)
    right_tree = build_tree(right_X, right_y, depth + 1, max_depth, min_samples_split, min_samples_leaf)

    return {
        'feature': feature_idx,
        'value': value,
        'left': left_tree,
        'right': right_tree
    }
```

build_tree(X, y): A recursive function implementing the decision tree learning algorithm (Hunt’s algorithm).
The function terminates if all labels are the same, if the number of samples is too small, or if the maximum depth is reached.
Otherwise, it selects the best feature and value to split the data, and recursively builds the left and right subtrees.
The tree is returned as a dictionary.

### Prediction Function

```bash

def predict(tree, X):
    if 'class' in tree:
        return tree['class']

    feature_value = X[tree['feature']]
    if feature_value <= tree['value']:
        return predict(tree['left'], X)
    else:
        return predict(tree['right'], X)
```

predict(tree, X): Recursively predicts the class for a given input (X) by traversing the decision tree.
If the node is a leaf (i.e., contains a class label), it returns that class.
Otherwise, it checks the value of the feature at the current node and traverses the left or right child accordingly.


### Metric Calculation Functions

```bash

def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def calculate_precision(y_true, y_pred, pos_label=1):
    true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == pos_label)
    predicted_positive = sum(1 for pred in y_pred if pred == pos_label)
    return true_positive / predicted_positive if predicted_positive else 0

def calculate_recall(y_true, y_pred, pos_label=1):
    true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == pos_label)
    actual_positive = sum(1 for true in y_true if true == pos_label)
    return true_positive / actual_positive if actual_positive else 0

def calculate_f1_score(y_true, y_pred, pos_label=1):
    precision = calculate_precision(y_true, y_pred, pos_label)
    recall = calculate_recall(y_true, y_pred, pos_label)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0
```

calculate_accuracy: Computes the accuracy of predictions by dividing the number of correct predictions by the total number of predictions.
calculate_precision: Measures the proportion of true positives out of all predicted positives.
calculate_recall: Measures the proportion of true positives out of all actual positives.
calculate_f1_score: Combines precision and recall into a harmonic mean to provide a balanced metric.

### Data Loading and Preprocessing

```bash

def load_data(file_path):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    data = []

    for row in sheet.iter_rows(values_only=True):
        data.append(row)

    X = [[float(val) for val in row[:-1]] for row in data if row[-1] is not None]
    y = [int(row[-1]) for row in data if row[-1] is not None]
    return X, y

```

load_data(file_path): Loads data from an Excel file using openpyxl.
The features (X) are all columns except the last one, and the target variable (y) is the last column.

### Data Splitting and Scaling

```bash

def train_test_split(X, y, test_size=0.3):
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

def min_max_scale(X):
    min_vals = [min(column) for column in zip(*X)]
    max_vals = [max(column) for column in zip(*X)]
    X_scaled = [[(X[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j]) if max_vals[j] - min_vals[j] != 0 else 0
                 for j in range(len(X[0]))] for i in range(len(X))]
    return X_scaled
```

train_test_split: Splits the dataset into training and testing sets based on the specified test size.
min_max_scale: Scales the features of X using Min-Max scaling so that all feature values lie between 0 and 1.


### Example Execution

```bash

file_path = r'H:\lesson\highest bachelor\Term 1\Data mining\EX\Serial1_EX\EX2\Train(2).xlsx'
X, y = load_data(file_path)

X_scaled = min_max_scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

tree = build_tree(X_train, y_train, max_depth=5, min_samples_split=5, min_samples_leaf=3)
y_pred = [predict(tree, x) for x in X_test]

accuracy = calculate_accuracy(y_test, y_pred)
precision = calculate_precision(y_test, y_pred)
recall = calculate_recall(y_test, y_pred)
f1 = calculate_f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
```
The example executes the decision tree classification on a dataset loaded from an Excel file, scales the features, splits the data, builds the decision tree, makes predictions, and then calculates and prints accuracy, precision, recall, and F1 score.


# Decision Tree Classifier from Scratch

This Python code implements a Decision Tree classifier using the Gini Impurity criterion for splitting. The classifier is trained on data loaded from an Excel file and is evaluated using accuracy, precision, recall, and F1 score metrics.

## Key Features:
- Gini Impurity for decision tree splits
- Recursive tree-building using Hunt's algorithm
- Prediction function for classifying new instances
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score
- Data loading from Excel files using `openpyxl`
- Min-Max scaling for feature normalization

## Functions:

### 1. `gini_impurity(y)`
Calculates the Gini Impurity for a given set of target labels (`y`).

### 2. `best_split(X, y)`
Finds the best feature and value to split the dataset using the Gini Impurity criterion.

### 3. `build_tree(X, y, ...)`
Recursively builds a decision tree using the best splits. Stops if all samples belong to the same class or other stopping criteria are met.

### 4. `predict(tree, X)`
Predicts the class for a given input `X` by traversing the decision tree.

### 5. Metric Functions:
- `calculate_accuracy(y_true, y_pred)`
- `calculate_precision(y_true, y_pred, pos_label=1)`
- `calculate_recall(y_true, y_pred, pos_label=1)`
- `calculate_f1_score(y_true, y_pred, pos_label=1)`

### 6. `load_data(file_path)`
Loads data from an Excel file and separates features and the target variable.

### 7. `train_test_split(X, y, test_size=0.3)`
Splits the dataset into training and testing sets.

### 8. `min_max_scale(X)`
Scales the features of the dataset using Min-Max scaling.

## Usage:
1. Provide the path to your Excel file with data.
2. The script will load the data, scale it, split it into training and testing sets, train a decision tree, and then evaluate the model.

### Example:

```python

file_path = "your_file_path.xlsx"
X, y = load_data(file_path)
X_scaled = min_max_scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

tree = build_tree(X_train, y_train, max_depth=5, min_samples_split=5, min_samples_leaf=3)
y_pred = [predict(tree, x) for x in X_test]

accuracy = calculate_accuracy(y_test, y_pred)
precision = calculate_precision(y_test, y_pred)
recall = calculate_recall(y_test, y_pred)
f1 = calculate_f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

```

# @Author : babak yousefian


---


# 3.EX2 part two is : 

# Detailed Explanation:

### 1. Entropy Function:

```bash

import math
import openpyxl

def entropy(y):
    total = len(y)
    class_counts = {}
    for label in y:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    entropy_val = 0.0
    for count in class_counts.values():
        prob = count / total
        entropy_val -= prob * math.log2(prob) if prob > 0 else 0
    return entropy_val
```

Imports:
math: Used for mathematical functions like log2.
openpyxl: Used to read data from Excel files.
entropy(y) function:
Purpose: Calculates the entropy of a dataset. Entropy is a measure of impurity or disorder, used in decision trees to decide the best splits.
y is the list of target labels (class labels).
It counts the occurrences of each class label, computes the probabilities, and applies the entropy formula:
Entropy=−∑p(x)log⁡2(p(x))
Entropy=−∑p(x)log2​(p(x)) where p(x)p(x) is the probability of each class.

### 2. Best Split Function:

```bash

def best_split(X, y):
    best_entropy = float('inf')
    best_split_feature = None
    best_split_value = None

    for feature_idx in range(len(X[0])):
        unique_values = sorted(set(row[feature_idx] for row in X))

        for value in unique_values:
            left_y = [y[i] for i in range(len(X)) if X[i][feature_idx] <= value]
            right_y = [y[i] for i in range(len(X)) if X[i][feature_idx] > value]

            left_entropy = entropy(left_y)
            right_entropy = entropy(right_y)
            weighted_entropy = (len(left_y) * left_entropy + len(right_y) * right_entropy) / len(y)

            if weighted_entropy < best_entropy:
                best_entropy = weighted_entropy
                best_split_feature = feature_idx
                best_split_value = value

    return best_split_feature, best_split_value
```

Purpose: Finds the best feature and value to split the dataset using the entropy measure.
Iterates over each feature, calculates the entropy for all possible splits, and chooses the split with the lowest weighted entropy (i.e., the most informative split).

### 3. Build Tree (Recursive Function - Hunt's Algorithm):

```bash

def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    if len(set(y)) == 1 or len(X) < min_samples_split or (max_depth is not None and depth >= max_depth):
        return {'class': max(set(y), key=y.count)}

    feature_idx, value = best_split(X, y)

    if feature_idx is None:
        return {'class': max(set(y), key=y.count)}

    left_X, left_y, right_X, right_y = [], [], [], []
    for i in range(len(X)):
        if X[i][feature_idx] <= value:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])

    left_tree = build_tree(left_X, left_y, depth + 1, max_depth, min_samples_split, min_samples_leaf)
    right_tree = build_tree(right_X, right_y, depth + 1, max_depth, min_samples_split, min_samples_leaf)

    return {
        'feature': feature_idx,
        'value': value,
        'left': left_tree,
        'right': right_tree
    }
```

Purpose: Recursively builds a decision tree using Hunt’s algorithm.
Base case:
If all labels are the same or if stopping conditions (like minimum samples or max depth) are met, return a leaf node with the most common class.
Otherwise:
Use best_split() to find the feature and value to split on.
Recursively build left and right subtrees for the resulting splits.

### 4. Predict Function:

```bash

def predict(tree, X):
    if 'class' in tree:
        return tree['class']

    feature_value = X[tree['feature']]
    if feature_value <= tree['value']:
        return predict(tree['left'], X)
    else:
        return predict(tree['right'], X)
```

Purpose: Predicts the class label for a given sample X using the decision tree.
If the tree is a leaf (i.e., it contains a class label), return that class.
Otherwise, compare the feature value to the split value and recursively predict using either the left or right subtree.

### 5. Metrics Functions (Accuracy, Precision, Recall, F1 Score):

```bash

def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def calculate_precision(y_true, y_pred, pos_label=1):
    true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == pos_label)
    predicted_positive = sum(1 for pred in y_pred if pred == pos_label)
    return true_positive / predicted_positive if predicted_positive else 0

def calculate_recall(y_true, y_pred, pos_label=1):
    true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == pos_label)
    actual_positive = sum(1 for true in y_true if true == pos_label)
    return true_positive / actual_positive if actual_positive else 0

def calculate_f1_score(y_true, y_pred, pos_label=1):
    precision = calculate_precision(y_true, y_pred, pos_label)
    recall = calculate_recall(y_true, y_pred, pos_label)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0
```

These functions calculate the standard classification metrics:
Accuracy: Proportion of correct predictions.
Precision: Proportion of true positives among predicted positives.
Recall: Proportion of true positives among actual positives.
F1 Score: Harmonic mean of precision and recall.


### 6. Load Data from Excel:


```bash

def load_data(file_path):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    data = []

    for row in sheet.iter_rows(values_only=True):
        data.append(row)

    X = [[float(val) for val in row[:-1]] for row in data if row[-1] is not None]
    y = [int(row[-1]) for row in data if row[-1] is not None]

    return X, y
```

Purpose: Loads data from an Excel file and separates it into features (X) and target labels (y).
It assumes that the target variable is in the last column of the Excel file.


### 7. Train-Test Split:

```bash

def train_test_split(X, y, test_size=0.3):
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
```

Purpose: Splits the dataset into training and test sets. The test_size parameter controls the proportion of data used for testing.


### 8. Min-Max Scaling:

```bash

def min_max_scale(X):
    min_vals = [min(column) for column in zip(*X)]
    max_vals = [max(column) for column in zip(*X)]
    X_scaled = [[(X[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j]) if max_vals[j] - min_vals[j] != 0 else 0
                 for j in range(len(X[0]))] for i in range(len(X))]
    return X_scaled
```

Purpose: Applies Min-Max scaling to normalize features. The scaling ensures all features lie between 0 and 1.


### 9. Example Execution:


```bash

file_path = r'H:\lesson\highest bachelor\Term 1\Data mining\EX\Serial1_EX\EX2\Train(2).xlsx'
X, y = load_data(file_path)

X_scaled = min_max_scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

tree = build_tree(X_train, y_train, max_depth=5, min_samples_split=5, min_samples_leaf=3)
y_pred = [predict(tree, x) for x in X_test]

accuracy = calculate_accuracy(y_test, y_pred)
precision = calculate_precision(y_test, y_pred)
recall = calculate_recall(y_test, y_pred)
f1 = calculate_f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

```
Purpose: This section executes the entire process:
Loads the data.
Scales the features.
Splits the data into training and testing sets.
Builds the decision tree on the training set.
Makes predictions on the test set.
Calculates and prints the classification metrics.


# Decision Tree Implementation in Python

## Overview
This Python script implements a Decision Tree classifier using recursive functions based on Hunt's Algorithm. It includes functions for data preprocessing, entropy calculation, tree building, prediction, and performance evaluation using common classification metrics (accuracy, precision, recall, and F1 score).

## Entropy Function
The `entropy()` function calculates the entropy of a dataset. Entropy is a measure of impurity or uncertainty. A lower entropy value indicates that the dataset is more homogenous.

## Best Split Function
The `best_split()` function identifies the feature and corresponding value that best divides the dataset into two subsets with minimal entropy. It iterates over all features and their possible split values to find the best split.

## Decision Tree Building (Hunt's Algorithm)
The `build_tree()` function recursively builds a decision tree. It continues splitting the data until certain stopping conditions are met (e.g., a single class label in a subset, reaching maximum depth, or insufficient data).

## Prediction Function
The `predict()` function traverses the decision tree to make predictions for unseen data. It compares the value of the feature at each node with the split value and moves down the appropriate branch.

## Evaluation Metrics
- `calculate_accuracy()`: Computes the proportion of correct predictions.
- `calculate_precision()`: Measures the proportion of true positives among predicted positives.
- `calculate_recall()`: Measures the proportion of true positives among actual positives.
- `calculate_f1_score()`: Harmonic mean of precision and recall.

## Data Preprocessing
- **Load Data**: The `load_data()` function loads the dataset from an Excel file.
- **Train-Test Split**: The `train_test_split()` function splits the dataset into training and testing sets.
- **Min-Max Scaling**: The `min_max_scale()` function normalizes the feature values to a range of 0 to 1.

## Example Execution
The script demonstrates the use of these functions by:
1. Loading the data from an Excel file.
2. Scaling the features using Min-Max scaling.
3. Splitting the data into training and testing sets.
4. Building a decision tree using the training set.
5. Evaluating the model's performance using the test set and printing the classification metrics (accuracy, precision, recall, and F1 score).

This markdown provides a complete overview of your code, including explanations of its key components and functionality.

---

# @Author : babak yousefian


---

# 4.EX3 is : 

# Explanation of Code

### 1. Importing Required Libraries

---

```bash
import numpy as np
import xlrd
from collections import Counter
```

numpy (imported as np): A package for numerical computations in Python, used here for array manipulation and mathematical operations.
xlrd: A library to read data from Excel files (.xls format). It helps in extracting data from Excel sheets.
collections.Counter: A class that counts the frequency of elements in an iterable. It's used for majority voting in KNN.

### 2. load_data(file_path) Function

```bash

def load_data(file_path):
    workbook = xlrd.open_workbook(file_path)
    sheet = workbook.sheet_by_index(0)

    data = []
    for row_idx in range(1, sheet.nrows):  # Skip header row
        row = sheet.row_values(row_idx)
        data.append(row)

    data_array = np.array(data)

    X = data_array[:, :-1]  # Features (all columns except the last)
    y = data_array[:, -1]  # Labels (last column)

    X = np.array(
        [[float(x) if isinstance(x, (int, float)) or x.replace('.', '', 1).isdigit() else 0 for x in row] for row in X])

    return X, y
```
   
Purpose: Loads the dataset from an Excel file, processes it, and splits it into feature data X and labels y.
Steps:
Open Excel file: Uses xlrd.open_workbook() to open the Excel file and reads the first sheet.
Extract rows: Loops over rows (excluding the header) and stores them as a list.
Split into features and labels: The features are all columns except the last, and labels are the last column.
Convert to numeric: Converts non-numeric values to 0. This is done by checking each entry and converting it to float if possible.

### 3. standardize(X) Function

```bash

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled
```

Purpose: Standardizes the feature matrix X using Z-score normalization.
Steps:
Calculate mean and standard deviation: For each feature (column) in X.
Apply Z-score formula: Subtract the mean and divide by the standard deviation to scale the data.


### 4. euclidean_distance(point1, point2) Function

```bash

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
```

Purpose: Computes the Euclidean distance between two points.
Steps:
Calculate the squared differences between each dimension of the two points.
Sum up the squared differences and take the square root to get the Euclidean distance.


### 5. knn(X_train, y_train, X_test, k=3) Function

```bash

def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        y_pred.append(most_common[0][0])

    return np.array(y_pred)
```

Purpose: Implements the K-Nearest Neighbors algorithm.
Steps:
Loop through test points: For each test point, calculate its distance from all training points.
Find the k nearest neighbors: Sort the distances and select the k smallest.
Majority voting: Use Counter to get the most common label among the k nearest neighbors.
Return predictions: After classifying all test points, return the predictions as a NumPy array.

### 6. k_fold_split(X, y, k=5) Function

```bash

def k_fold_split(X, y, k=5):
    indices = np.random.permutation(len(X))
    fold_size = len(X) // k
    folds = []

    for i in range(k):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        folds.append((X_train, y_train, X_test, y_test))

    return folds
```

Purpose: Splits the dataset into k folds for cross-validation.
Steps:
Shuffle indices: Randomly permutes the dataset indices to ensure random splits.
Create k folds: Divides the data into k folds (subsets) for cross-validation.
Split the data: For each fold, one part is used for testing, and the rest is used for training.

### 7. evaluate(y_test, y_pred) Function

```bash

def evaluate(y_test, y_pred):
    accuracy = np.mean(y_test == y_pred)

    labels = np.unique(y_test)

    precision = 0
    recall = 0
    f1 = 0
    for label in labels:
        true_positive = np.sum((y_test == label) & (y_pred == label))
        false_positive = np.sum((y_test != label) & (y_pred == label))
        false_negative = np.sum((y_test == label) & (y_pred != label))

        if true_positive + false_positive == 0:
            label_precision = 0
        else:
            label_precision = true_positive / (true_positive + false_positive)

        if true_positive + false_negative == 0:
            label_recall = 0
        else:
            label_recall = true_positive / (true_positive + false_negative)

        if label_precision + label_recall == 0:
            label_f1 = 0
        else:
            label_f1 = 2 * (label_precision * label_recall) / (label_precision + label_recall)

        precision += label_precision
        recall += label_recall
        f1 += label_f1

    precision /= len(labels)
    recall /= len(labels)
    f1 /= len(labels)

    return accuracy, precision, recall, f1
```

Purpose: Computes the evaluation metrics (accuracy, precision, recall, F1 score) for model performance.
Steps:
Accuracy: Proportion of correctly predicted labels.
Precision: Proportion of positive predictions that are actually positive.
Recall: Proportion of actual positives that are correctly predicted.
F1 Score: Harmonic mean of precision and recall.
For multi-class: Averages the metrics across all unique labels.

### 8. cross_validate(X, y, k_folds=5, knn_k=3) Function

```bash

def cross_validate(X, y, k_folds=5, knn_k=3):
    folds = k_fold_split(X, y, k=k_folds)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for fold in folds:
        X_train, y_train, X_test, y_test = fold

        X_train_scaled = standardize(X_train)
        X_test_scaled = standardize(X_test)

        y_pred = knn(X_train_scaled, y_train, X_test_scaled, k=knn_k)

        accuracy, precision, recall, f1 = evaluate(y_test, y_pred)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    return avg_accuracy, avg_precision, avg_recall, avg_f1
```


Purpose: Performs cross-validation on the dataset and returns average evaluation metrics.
Steps:
Split the data: Uses k_fold_split to create the data folds.
Standardize: Scales the training and testing data using standardize.
Train and predict: Uses the knn function to make predictions.
Evaluate: Computes accuracy, precision, recall, and F1 score.
Average metrics: Averages the results across all folds.

### 9. main(file_path) Function

```bash

def main(file_path):
    X, y = load_data(file_path)

    avg_accuracy, avg_precision, avg_recall, avg_f1 = cross_validate(X, y, k_folds=5, knn_k=3)

    print(f"Cross-Validation Results (5-fold):")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
```

Purpose: The main driver function that loads data, runs cross-validation, and prints results.
Steps:
Load data: Calls load_data to load the dataset.
Cross-validation: Performs cross-validation using the cross_validate function.
Print results: Displays the average evaluation metrics.


# K-Nearest Neighbors (KNN) Algorithm with Cross-Validation

## Overview
This Python project implements the K-Nearest Neighbors (KNN) algorithm with cross-validation on a dataset stored in an Excel file. The dataset is split into training and testing sets, and the model is evaluated using multiple performance metrics: Accuracy, Precision, Recall, and F1 Score.

## Libraries Used
- **`numpy`**: For numerical operations and array manipulations.
- **`xlrd`**: To read Excel files (.xls format).
- **`collections.Counter`**: For majority voting in KNN classification.

## Functions

### `load_data(file_path)`
Loads the dataset from an Excel file and splits it into feature data (`X`) and labels (`y`).

### `standardize(X)`
Standardizes the feature matrix using Z-score normalization.

### `euclidean_distance(point1, point2)`
Computes the Euclidean distance between two data points.

### `knn(X_train, y_train, X_test, k=3)`
Implements the KNN algorithm for classification, using the specified value of `k` for nearest neighbors.

### `k_fold_split(X, y, k=5)`
Splits the dataset into `k` folds for cross-validation.

### `evaluate(y_test, y_pred)`
Evaluates the model's performance using Accuracy, Precision, Recall, and F1 Score.

### `cross_validate(X, y, k_folds=5, knn_k=3)`
Performs cross-validation on the dataset and returns average evaluation metrics.

### `main(file_path)`
The main function that loads the dataset, runs cross-validation, and prints the average evaluation metrics.

## Usage
Run the script with the path to your Excel dataset:

```bash
python knn_model.py

Ensure the dataset has the features in columns and labels in the last column.
Conclusion

This project demonstrates the implementation of the KNN algorithm with cross-validation and evaluation metrics.
```

# @Author : babak yousefian

---
---
---

