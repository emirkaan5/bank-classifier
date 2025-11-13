import pandas as pd
import numpy as np
import re
from pathlib import Path


DATA_DIR = Path("data")
data_list = list(DATA_DIR.rglob("*.csv"))
for file in data_list:
    
    print("cleaning the data:  " +str(file))
    data = pd.read_csv(file)
    cleansed_df =pd.DataFrame().reindex_like(data)
    #cleanse descriptions
    data['description'] = data['description'].astype(str).str.replace(r"\d{1,2}\/\d{1,2}\/\d{2,4}", " ", regex=True)
    data.drop(columns= ['type','source_line','source_file','balance'],axis=1,inplace=True)

    print(data.columns)
    data.to_csv(f"cleansed_{str(file)}")


