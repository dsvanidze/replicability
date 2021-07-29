import numpy as np
import pandas as pd
from utils import ROOT_DIR

def describe_data():
    """Return Panda summary of data
    Examples
    --------
    >>> describe_data()
    """
    
    data_to_split = pd.read_csv(ROOT_DIR + "/data/csvs/data.csv", index_col="index")
    normalized_data = data_to_split.copy()
    normalized_data["LONG"] = normalized_data["LONG"] / 180
    normalized_data["LAT"] = normalized_data["LAT"] / 90
    normalized_data["LIR"] = np.log(normalized_data["IR"] + 1)
    normalized_data.pop("IR")
    normalized_data = normalized_data[["LONG", "LAT", "ACCESS", "PET", "POP", "URBAN", "WACCESS", "LIR"]]
    desc = normalized_data.describe().round(4)
    return desc
    