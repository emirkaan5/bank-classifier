import pandas as pd
from pathlib import Path

DATA_DIR = Path("cleansed_data")
OUT_DIR = Path("labeled")
OUT_DIR.mkdir(exist_ok=True)

label_enums = {
    '1': 'grocery',
    '2': 'dining',
    '3': 'travel',
    '4': 'shopping',
    '5': 'subscriptions',
    '6': 'utilities',
    '7': 'health',
    '8': 'entertainment',
    '9': 'other'
}

for file in DATA_DIR.glob("*.csv"):
    print(f"Labeling the file :   {file}")
    data = pd.read_csv(file)

    for idx, row in data.iterrows():
        while True:
            label = input(
                f"Label this data :\n"
                f"  {row['description']}, {row['amount']}\n"
                f"Labels:\n"
                f"  1. Grocery, 2. Dining, 3. Travel, 4. Shopping,\n"
                f"  5. Subscriptions, 6. Utilities, 7. Health,\n"
                f"  8. Entertainment, 9. Other\n> "
            )

            if label in label_enums:
                data.loc[idx, "label"] = label_enums[label]
                break
            else:
                print("Invalid label, try again.")

    save_path = OUT_DIR / file.name
    data.to_csv(save_path, index=False)
    print(f"{file} is labeled and saved in : {save_path}")
