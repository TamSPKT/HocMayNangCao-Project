from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from ClassifierHelpers import save_model


def print_elapsed_time(message="{0}", start_time: float | None = None):
    if start_time is not None:
        elapsed = time.perf_counter() - start_time
        print(message.format(timedelta(seconds=elapsed)))
    return time.perf_counter()


def get_columns_types(size: int = 128):
    columns_types: dict = {"Label": str}
    for i in range(size):
        for j in range(size):
            columns_types[f"{i:0>3d}:{j:0>3d}"] = np.uint8
            # Range np.uint8: From 0 to 255
    return columns_types
    # TypeError: descriptor '__array_wrap__' for 'numpy.generic' objects doesn't apply to a 'numpy.ndarray' object
    # return defaultdict(np.uint8, Label=str)


CSV_PATH = "./dataset-csv/images.csv"
SAVE_PATH = "./checkpoint-csv"

le = LabelEncoder()
# NOTE: Set n_components as int to speed up pca.fit_transform()
pca = PCA(n_components=256)
# pca = IncrementalPCA(n_components=256)
model = HistGradientBoostingClassifier(verbose=2)
# model = DecisionTreeClassifier()

start_time = print_elapsed_time()
df = pd.read_csv(CSV_PATH, dtype=get_columns_types(size=32))  # 134274 lines: 274.01 MB
df.info(verbose=False, memory_usage="deep")  # 134273 entries: 138.6 MB
start_time = print_elapsed_time(
    "- Finished loading DataFrame from CSV file in {0}", start_time
)  # 134273 entries: About 9 minutes

# df["Label"] = df["Label"].astype("str")
# NOTE: astype("category") will cause model.fit(...) ValueError
labels = df["Label"].copy()
print(f"+ Labels shape: {labels.shape}")
df = df.drop(columns="Label")
# NOTE: Mapping back to original values in Dataframe elementwise is time-consuming (Still running after about 30 minutes)
# df = df.map(lambda x: np.uint8(abs(255 - x)))  # Inverse back to original values
start_time = print_elapsed_time(
    "- Finished preparing DataFrame for training in {0}", start_time
)

values_labels = le.fit_transform(labels)
# values_labels = lb.fit_transform(labels)
values_pca = pca.fit_transform(df)
print(f"+ PCA reduction: {df.shape} => {values_pca.shape}")
start_time = print_elapsed_time("- Finished LabelEncoder & PCA in {0}", start_time)

model = model.fit(X=values_pca, y=values_labels)
start_time = print_elapsed_time("- Finished fitting model in {0}", start_time)

# with pd.read_csv(CSV_PATH, iterator=True, chunksize=1000) as reader:
#     for chunk in reader:
#         # print(type(chunk))
#         df: pd.DataFrame = chunk
#         # df.info(verbose=False, memory_usage="deep") # 125MB

#         labels = df["Label"].copy().astype("category")
#         df = df.drop(columns="Label")

#         values_ipca = ipca.fit_transform(df)
#         model = model.fit(X=values_ipca, y=labels)

save_model(SAVE_PATH, labelEncoder=le, pca=pca, classifier=model)

print("Done!")
