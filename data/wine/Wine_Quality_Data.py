import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
from sklearn import preprocessing

from typing import Tuple

import pathlib


WINE_DATA_WHITE_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), "winequality-white.csv")
WINE_DATA_RED_PAT = os.path.join(pathlib.Path(__file__).parent.absolute(), "winequality-red.csv")

def import_wine_dataset() -> pd.DataFrame:
    white_wine = pd.read_csv(WINE_DATA_WHITE_PATH, sep=";")
    red_wine = pd.read_csv(WINE_DATA_RED_PAT, sep=";")
    wine = white_wine.append(red_wine, ignore_index=True)
    return wine


def change_labels_low_avg_high(wine: pd.DataFrame) -> pd.DataFrame:
    # Convert Low Average High
    wine.quality.values[wine.quality.values < 6] = 0
    wine.quality.values[wine.quality.values == 6] = 1
    wine.quality.values[wine.quality.values > 6] = 2

    # Distinguish only between average and high quality
    wine = wine[wine.quality != 0]
    wine.quality.values[wine.quality.values == 1] = 0
    wine.quality.values[wine.quality.values == 2] = 1
    # Get even distribution
    test = wine.groupby('quality').apply(lambda x: x.sample(min(wine.groupby('quality')['quality'].count()))).reset_index(drop=True)
    return test


def get_normalized_train_test_split(wine: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = wine["quality"].copy()
    features = wine.drop(["quality"], axis=1)

    # Test Train Split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)
    NORMALIZE = True
    MIN_MAX = True
    if NORMALIZE:
        if MIN_MAX:
            scaler = preprocessing.MinMaxScaler()
        else:
            scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)

        X_test_scaled = scaler.fit_transform(X_test.values)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train, X_test, y_train, y_test


def get_wine_dataset():
    X_train, X_test, y_train, y_test = get_normalized_train_test_split(change_labels_low_avg_high(import_wine_dataset()))
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    get_wine_dataset()
