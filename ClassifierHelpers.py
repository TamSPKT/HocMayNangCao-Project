from datetime import datetime
import os
import joblib
from sklearn.base import ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import HistGradientBoostingClassifier


def load_classifier(path: str):
    classifier: HistGradientBoostingClassifier = joblib.load(path)
    return classifier


def load_pca(path: str):
    pca: PCA | IncrementalPCA = joblib.load(path)
    return pca


def load_encoder(path: str):
    encoder: LabelEncoder = joblib.load(path)
    return encoder

def save_model(
    path: str,
    labelEncoder: LabelEncoder,
    pca: PCA | IncrementalPCA,
    classifier: ClassifierMixin,
):
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    path = os.path.normpath(path)
    print(
        "+ Save LabelEncoder:",
        joblib.dump(
            labelEncoder,
            os.path.join(path, f"{type(labelEncoder).__name__}-{dt}.pkl"),
        ),
    )
    print(
        "+ Save PCA:",
        joblib.dump(pca, os.path.join(path, f"{type(pca).__name__}-{dt}.pkl")),
    )
    print(
        "+ Save Classifier:",
        joblib.dump(
            classifier, os.path.join(path, f"{type(classifier).__name__}-{dt}.pkl")
        ),
    )
