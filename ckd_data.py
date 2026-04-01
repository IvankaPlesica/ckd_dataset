"""
CKD dataset loading and cleaning.
"""

import pandas as pd

def load_ckd():
    df = pd.read_csv("downloads-cache/uciml/336.csv.gz")
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df["dm"]    = df["dm"].replace("\tno", "no")
    df["class"] = df["class"].replace("ckd\t", "ckd")
    return df