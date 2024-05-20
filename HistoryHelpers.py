import os
import joblib
from keras.callbacks import History
import pandas as pd

SAVE_PATH = "./logs"


def save_history(history: History, timestamp: str):
    print(f"- History params: {history.params}")
    # file_path = os.path.normpath(os.path.join(SAVE_PATH, f"history-{timestamp}.pkl"))
    file_path = os.path.normpath(os.path.join(SAVE_PATH, f"history-{timestamp}.csv"))

    # Convert the history.history dict to a pandas DataFrame:
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(file_path)
    # joblib.dump(history, filename=file_path)
    print(f"+ Saved History: {file_path}")
