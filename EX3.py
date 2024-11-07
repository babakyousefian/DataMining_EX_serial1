import numpy as np
import xlrd
from collections import Counter


# Function to load the dataset from an Excel file
def load_data(file_path):
    # Open the Excel file and read the sheet
    workbook = xlrd.open_workbook(file_path)
    sheet = workbook.sheet_by_index(0)

    # Extract the data from the sheet (assuming the last column is the label)
    data = []
    for row_idx in range(1, sheet.nrows):  # Skip header row
        row = sheet.row_values(row_idx)
        data.append(row)

    # Convert to numpy array for easier manipulation
    data_array = np.array(data)

    # Ensure all feature data (X) is numeric
    X = data_array[:, :-1]  # Features (all columns except the last)
    y = data_array[:, -1]  # Labels (last column)

    # Convert X to numeric values (if possible)
    X = np.array(
        [[float(x) if isinstance(x, (int, float)) or x.replace('.', '', 1).isdigit() else 0 for x in row] for row in X])

    return X, y


# Standardize the features (Z-score normalization)
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled


# Function to compute the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# KNN Algorithm Implementation
def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        # Calculate the distance from this test point to all training points
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]

        # Get the indices of the k closest points
        k_indices = np.argsort(distances)[:k]

        # Get the labels of the k closest points
        k_nearest_labels = [y_train[i] for i in k_indices]

        # Majority voting: Get the most common label among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        y_pred.append(most_common[0][0])

    return np.array(y_pred)


# Split the dataset into k folds
def k_fold_split(X, y, k=5):
    # Generate k-fold splits
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


# Evaluation metrics (Accuracy, Precision, Recall, F1 Score)
def evaluate(y_test, y_pred):
    accuracy = np.mean(y_test == y_pred)

    # Precision, Recall, F1 (for binary classification or multi-class, weighted average)
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


# Cross-validation function to evaluate KNN model
def cross_validate(X, y, k_folds=5, knn_k=3):
    folds = k_fold_split(X, y, k=k_folds)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for fold in folds:
        X_train, y_train, X_test, y_test = fold

        # Standardize the features
        X_train_scaled = standardize(X_train)
        X_test_scaled = standardize(X_test)

        # Train the KNN model and make predictions
        y_pred = knn(X_train_scaled, y_train, X_test_scaled, k=knn_k)

        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate(y_test, y_pred)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Calculate the average of all metrics across the folds
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    return avg_accuracy, avg_precision, avg_recall, avg_f1


# Main function to run the KNN algorithm with cross-validation
def main(file_path):
    # Load the dataset
    X, y = load_data(file_path)

    # Cross-validate the model
    avg_accuracy, avg_precision, avg_recall, avg_f1 = cross_validate(X, y, k_folds=5, knn_k=3)

    # Print the average evaluation metrics
    print(f"Cross-Validation Results (5-fold):")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")


# Call the main function with the path to your dataset
if __name__ == "__main__":
    file_path = r'H:\lesson\highest bachelor\Term 1\Data mining\EX\Serial1_EX\EX3\dataset3.xls'
    main(file_path)
