import pandas as pd
from pathlib import Path

DATA_DIR = Path("cleansed_data")
data_list = list(DATA_DIR.rglob("*.csv"))
for file in DATA_DIR.iterdir():
    data = pd.read_csv(file)
    for idx, row in data.iterrows():
        label = input(f"Label this data :\n {row['description']}, {row['amount']} \n Labels:\n  1. Grocery,2. Dining, 3. Travel, 4. Shopping, 5. Subscriptions, 6. Utilities, 7. Health,. 8. Entertainment, 9. other\n ")
        
        row['label'] = label

    save_dir = "labeled/"+str(file.split("/"))[1]
    data_to_csv(save_dir)
    print(f"{file} is labeled and saved in : {save_dir}")
