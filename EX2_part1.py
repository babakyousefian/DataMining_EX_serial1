import openpyxl

# Gini Impurity Function
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


# Function to Find the Best Split
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


# Recursive Function for Decision Tree (Hunt's Algorithm)
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


# Function to Predict with the Decision Tree
def predict(tree, X):
    if 'class' in tree:
        return tree['class']

    feature_value = X[tree['feature']]
    if feature_value <= tree['value']:
        return predict(tree['left'], X)
    else:
        return predict(tree['right'], X)


# Functions to Calculate Metrics Manually
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


# Load and Process Data from Excel Manually
def load_data(file_path):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    data = []

    # Iterate through rows and cells to collect data
    for row in sheet.iter_rows(values_only=True):
        data.append(row)

    # Separate features and target variable
    X = [[float(val) for val in row[:-1]] for row in data if row[-1] is not None]
    y = [int(row[-1]) for row in data if row[-1] is not None]
    return X, y


# Split Data
def train_test_split(X, y, test_size=0.3):
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


# Min-Max Scaling
def min_max_scale(X):
    min_vals = [min(column) for column in zip(*X)]
    max_vals = [max(column) for column in zip(*X)]
    X_scaled = [[(X[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j]) if max_vals[j] - min_vals[j] != 0 else 0
                 for j in range(len(X[0]))] for i in range(len(X))]
    return X_scaled


# Example Execution
# File path to your Excel file
file_path = r'H:\lesson\highest bachelor\Term 1\Data mining\EX\Serial1_EX\EX2\Train(2).xlsx'
X, y = load_data(file_path)

X_scaled = min_max_scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

tree = build_tree(X_train, y_train, max_depth=5, min_samples_split=5, min_samples_leaf=3)
y_pred = [predict(tree, x) for x in X_test]

# Metrics Calculation
accuracy = calculate_accuracy(y_test, y_pred)
precision = calculate_precision(y_test, y_pred)
recall = calculate_recall(y_test, y_pred)
f1 = calculate_f1_score(y_test, y_pred)

# Print the Metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
