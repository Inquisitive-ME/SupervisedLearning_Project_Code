import numpy as np
from sklearn.datasets import make_classification
import sklearn
import sklearn.model_selection

def generate_noisy_nonlinear_dataset():
    n_features = 10
    n_informative = 10
    n_redundant = 0
    n_clusters_per_class = 4
    hypercube = True
    random_state = 7
    x, y = make_classification(n_samples=3500, n_features=n_features, n_redundant=n_redundant,
                               n_informative=n_informative, n_classes=2, n_clusters_per_class=n_clusters_per_class,
                               flip_y=0.3, hypercube=hypercube, random_state=random_state, class_sep=4)
    x = np.apply_along_axis(lambda x: 2 ** x, 0, x)
    x = x / x.max(axis=0)
    return x,y

def get_noisy_nonlinear():
    X2, Y2 = generate_noisy_nonlinear_dataset()

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X2, Y2, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def generate_large_num_features_dataset():
    n_features = 200
    n_informative = 50
    n_redundant = 50
    n_clusters_per_class = 4
    hypercube = True
    X2, Y2 = make_classification(n_samples=3500, n_features=n_features, n_redundant=n_redundant,
                                 n_informative=n_informative, n_classes=2, n_clusters_per_class=n_clusters_per_class,
                                 flip_y=0.0, hypercube=hypercube, class_sep=1)
    return X2, Y2

def get_large_num_features_dataset():
    X2, Y2 = generate_large_num_features_dataset()

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X2, Y2, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test