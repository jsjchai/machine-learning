import os.path

import pandas as pd

from housing.download import HOUSING_PATH


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    print("csv文件：" + csv_path)
    return pd.read_csv(csv_path)
