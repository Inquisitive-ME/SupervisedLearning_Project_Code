import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib
import os


def perform_grid_search(parameters, X_train, y_train, GS_FILE_NAME_PREFIX):
    GS_FILE_NAME = GS_FILE_NAME_PREFIX
    for key, value in parameters.items():
        GS_FILE_NAME += ("_" + key)
    GS_FILE_NAME += ".pickle"

    if os.path.exists(GS_FILE_NAME):
        print("WARNING: file ", GS_FILE_NAME, " already exists")
        print("NOT performing Grid Search")
        gs = joblib.load(GS_FILE_NAME)
    else:
        print("Grid Search Will be Saved to ", GS_FILE_NAME)

        gs = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy'), parameters, return_train_score=True,
                          verbose=10, n_jobs=-1)
        gs.fit(X_train, y_train)

        joblib.dump(gs, GS_FILE_NAME)
        print("Saved ", GS_FILE_NAME)
    return gs