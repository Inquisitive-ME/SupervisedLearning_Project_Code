from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os


def perform_grid_search(parameters, X_train, y_train, scoring, GS_FILE_NAME_PREFIX, default_parameters={}):
    GS_FILE_NAME = GS_FILE_NAME_PREFIX
    for key, value in parameters.items():
        GS_FILE_NAME += ("_" + key)
        try:
            GS_FILE_NAME += ("_" + str(value[0]) + "-" + str(value[-1]))
        except ValueError:
            GS_FILE_NAME += ("_" + value[0] + "-" + value[-1])
    GS_FILE_NAME += ".pickle"

    if os.path.exists(GS_FILE_NAME):
        print("WARNING: file ", GS_FILE_NAME, " already exists")
        print("NOT performing Grid Search")
        gs = joblib.load(GS_FILE_NAME)
    else:
        print("Grid Search Will be Saved to ", GS_FILE_NAME)

        gs = GridSearchCV(AdaBoostClassifier(**default_parameters), parameters, scoring=scoring, return_train_score=True,
                          verbose=10, n_jobs=-1)
        gs.fit(X_train, y_train)

        joblib.dump(gs, GS_FILE_NAME)
        print("Saved ", GS_FILE_NAME)
    return gs