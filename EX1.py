import pandas as pd
import numpy as np
import xlrd
import math
from collections import Counter


# Load data from the XLS file
file_path = r'H:\lesson\highest bachelor\Term 1\Data mining\EX\Serial1_EX\EX1\dataset2.xls'
data = pd.read_excel(file_path, header=None, engine="xlrd")

# Convert the DataFrame to a numpy array (assuming each row is a data point)
data_array = data.to_numpy()

workbook = xlrd.open_workbook(file_path)
sheet = workbook.sheet_by_index(0)  # Read the first sheet

# Convert the sheet data into a list of lists (rows and columns)
data1 = []
for row_idx in range(sheet.nrows):
    row = sheet.row_values(row_idx)
    data1.append(row)


# Function to calculate entropy of a single feature (column)
def entropy(values):
    # Count frequency of each unique value
    value_counts = Counter(values)
    total_values = len(values)

    # Calculate entropy using the formula: H(X) = -Σ P(x_i) * log2(P(x_i))
    ent = 0
    for count in value_counts.values():
        probability = count / total_values
        if probability > 0:
            ent -= probability * math.log2(probability)

    return ent


# Function to calculate entropy for each feature in the dataset
def calculate_entropy_for_features(data1):
    entropies = []
    num_columns = len(data1[0])  # Number of columns (features)

    # Iterate over each feature (column)
    for col_idx in range(num_columns):
        feature_values = [data1[row_idx][col_idx] for row_idx in range(len(data1))]
        entropies.append(entropy(feature_values))  # Calculate entropy for each feature

    return entropies


# Calculate the Euclidean distance matrix
def euclidean_distance_matrix(data):
    num_points = data.shape[0]
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.sqrt(np.sum((data[i] - data[j]) ** 2))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix

    return distance_matrix


# Function to calculate the covariance manually between two variables
def covariance(X, Y):
    num_samples = len(X)
    mean_X = sum(X) / num_samples
    mean_Y = sum(Y) / num_samples
    cov = sum((X[i] - mean_X) * (Y[i] - mean_Y) for i in range(num_samples)) / (num_samples - 1)
    return cov


# Function to calculate the covariance matrix (manual implementation)
def covariance_matrix(data):
    num_samples = len(data)
    num_features = len(data[0])  # Number of features (columns)

    # Initialize the covariance matrix
    cov_matrix = [[0] * num_features for _ in range(num_features)]

    # Calculate the covariance for each pair of features (i, j)
    for i in range(num_features):
        for j in range(num_features):
            cov_matrix[i][j] = covariance([data[k][i] for k in range(num_samples)], [data[k][j] for k in range(num_samples)])

    return cov_matrix


# Function to compute the inverse of a matrix using Gaussian elimination (manual method)
def inverse_matrix(matrix):
    n = len(matrix)

    # Manually create an identity matrix of size n
    identity_matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    # Manually create the augmented matrix [A|I]
    augmented_matrix = [matrix[i] + identity_matrix[i] for i in range(n)]

    # Apply Gaussian elimination
    for i in range(n):
        # Make the diagonal element to 1
        diagonal_element = augmented_matrix[i][i]
        augmented_matrix[i] = [x / diagonal_element for x in augmented_matrix[i]]

        for j in range(n):
            if i != j:
                # Subtract the appropriate multiple of row i from row j
                factor = augmented_matrix[j][i]
                augmented_matrix[j] = [
                    augmented_matrix[j][k] - factor * augmented_matrix[i][k] for k in range(2 * n)
                ]

    # Extract the right half of the augmented matrix, which is the inverse
    return [row[n:] for row in augmented_matrix]


# Function to calculate the Mahalanobis distance between two vectors
def mahalanobis_distance(x, y, inv_cov_matrix):
    # Compute the difference between the vectors
    diff = [xi - yi for xi, yi in zip(x, y)]  # Element-wise difference between x and y

    # Step 1: Calculate the intermediate result by multiplying diff and the inverse covariance matrix (matrix-vector multiplication)
    intermediate = []
    for i in range(len(inv_cov_matrix)):  # For each row in the inverse covariance matrix
        row_result = sum(diff[j] * inv_cov_matrix[i][j] for j in range(len(diff)))  # Dot product of row and diff
        intermediate.append(row_result)

    # Step 2: Calculate the final dot product between intermediate and the diff vector
    mahalanobis_dist = sum(intermediate[i] * diff[i] for i in range(len(diff)))  # Final dot product

    # Return the square root of the result
    return mahalanobis_dist ** 0.5


# Function to calculate the Mahalanobis distance matrix (manual implementation)
def mahalanobis_distance_matrix(data):
    # Calculate covariance matrix
    cov_matrix = covariance_matrix(data)

    # Calculate inverse covariance matrix
    inv_cov_matrix = inverse_matrix(cov_matrix)

    # Initialize an empty distance matrix
    num_points = len(data)
    dist_matrix = np.zeros((num_points, num_points))

    # Calculate pairwise Mahalanobis distances
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = mahalanobis_distance(data[i], data[j], inv_cov_matrix)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Symmetric matrix

    return dist_matrix


# Function to calculate the correlation matrix
def correlation_matrix(data):
    num_features = len(data[0])  # Number of features (columns)

    # Calculate covariance matrix
    cov_matrix = covariance_matrix(data)

    # Initialize correlation matrix
    corr_matrix = [[0] * num_features for _ in range(num_features)]

    # Calculate correlation matrix using the formula: correlation = cov(X, Y) / (std(X) * std(Y))
    for i in range(num_features):
        for j in range(num_features):
            # Calculate standard deviations for each feature
            std_i = (sum((data[k][i] - sum([data[k][i] for k in range(len(data))]) / len(data)) ** 2 for k in range(len(data))) / len(data)) ** 0.5
            std_j = (sum((data[k][j] - sum([data[k][j] for k in range(len(data))]) / len(data)) ** 2 for k in range(len(data))) / len(data)) ** 0.5
            corr_matrix[i][j] = cov_matrix[i][j] / (std_i * std_j)  # Normalize covariance to get correlation

    return corr_matrix


# Generate the Euclidean distance matrix
distance_matrix = euclidean_distance_matrix(data_array)

# Print the Euclidean distance matrix
print("\n\n-------------------------------------------------------------------------------")
print("Euclidean Distance Matrix:")
print("------------------------------------------------------------------------------- \n\n ")
print(distance_matrix)

# Generate the Mahalanobis distance matrix
mahalanobis_matrix = mahalanobis_distance_matrix(data_array)

# Print the Mahalanobis distance matrix
print("\n\n-------------------------------------------------------------------------------")
print("Mahalanobis Distance Matrix:")
print("------------------------------------------------------------------------------- \n\n ")
print(mahalanobis_matrix)

# Generate the Correlation matrix
correlation_matrix_result = correlation_matrix(data_array)

# Print the Correlation matrix as plain floats
print("\n\n-------------------------------------------------------------------------------")
print("Correlation Matrix:")
print("------------------------------------------------------------------------------- \n\n ")
for row in correlation_matrix_result:
    print([float(value) for value in row])  # Convert np.float64 to plain float


# Calculate entropy for each feature
entropies = calculate_entropy_for_features(data1)

print("\n\n-------------------------------------------------------------------------------")
print("Entropy is : ")
print("------------------------------------------------------------------------------- \n\n ")
# Output entropy for each feature
for i, entropy_value in enumerate(entropies):
    print(f"Entropy of feature {i + 1}: {entropy_value}")

print("\n\n------------------------------------------------------------------------------- \n\n ")
