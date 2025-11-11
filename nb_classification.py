import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR= Path("data")


data = []
for file in DATA_DIR.iterdir():
    if file.is_file():
        file_data = pd.read_csv(file)
        data

