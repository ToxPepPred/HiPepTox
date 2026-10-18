import numpy as np
from sklearn.neighbors import NearestNeighbors


def SMOTE(X, y, N, k=3):
    # Separate the majority-class and minority-class samples
    X_majority = X[y == 0]
    X_minority = X[y == 1]

    # Calculate the number of synthetic samples to generate for each minority-class instance
    N_per_sample = N // len(X_minority)

    # If k exceeds the number of minority samples, reduce it to the maximum possible value
    k = min(k, len(X_minority) - 1)

    # Initialize a list to store synthetic samples and their corresponding labels
    synthetic_samples = []
    synthetic_labels = []

    # Fit a k-nearest neighbors (KNN) model on the minority class samples
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_minority)

    for minority_sample in X_minority:
        # Find the k-nearest neighbors for the current minority class sample
        _, indices = knn.kneighbors(minority_sample.reshape(1, -1), n_neighbors=k)

        # Randomly select k neighbors and create synthetic samples
        for _ in range(N_per_sample):
            neighbor_index = np.random.choice(indices[0])
            neighbor = X_minority[neighbor_index]

            # Calculate the difference between the current minority-class sample and its neighbors
            difference = neighbor - minority_sample

            # Generate a random number between 0 and 1
            alpha = np.random.random()

            # Create a synthetic sample as a linear combination of the minority-class sample and its neighbor
            synthetic_sample = minority_sample + alpha * difference

            # Append the synthetic sample and its label to the list
            synthetic_samples.append(synthetic_sample)
            synthetic_labels.append(1)

    # Convert the list to a NumPy array
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.array(synthetic_labels)

    # Combine the original majority-class samples with the synthetic samples
    X_balanced = np.concatenate((X_majority, X_synthetic), axis=0)
    y_balanced = np.concatenate((np.zeros(len(X_majority)), y_synthetic), axis=0)

    return X_balanced, y_balanced


def datapro(x, y):
    xx = []
    for i in x:
        xx.append(i.flatten().tolist())
    xx = np.array(xx)
    yy = y.reshape(1, -1)[0]
    X_balanced, y_balanced = SMOTE(xx, yy, N=444)  # 1818
    X_train = np.concatenate((X_balanced, xx[:444]), axis=0)  # 1818
    y_train = np.concatenate((y_balanced, yy[:444]), axis=0)  # 1818
    XX = []
    for i in X_train:
        XX.append(i.reshape(49, 96))
    XX = np.array(XX)
    y_train = np.array([int(i) for i in y_train])
    y_train = y_train.reshape(1332, 1)  # 8205
    return XX, np.array(y_train)
