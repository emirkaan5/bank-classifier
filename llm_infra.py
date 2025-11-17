from transformers import pipeline
import torch

pipe = pipeline("text-generation", model="google/gemma-3-1b-it", device="cuda", torch_dtype=torch.bfloat16)

from pathlib import Path

DATA_DIR = Path("cleansed_data")
data_list = list(DATA_DIR.rglob("*.csv"))


prompt = """
Following text, with an amount of expense is from a bank statement document.
You are tasked to label this text into following 9 categories:
    1.Grocery
    2.Dining
    3.Travel
    4.Shopping
    5.Subscription
    6.Utilites
    7.Health
    8.Entertainment
    9.Other

If you are absolutely uncertain on which one to put one label in, put to 9.
You must always give a guess. Here is an example:

    DESCRIPTION: HANGAR PUB WINGS AMHER10 UNIVERSITY DR AMHERST 01002 MA USA 1%
    AMOUNT: 	50.44
    Label: Dining

Here is data to be labeled: 

    DESCRIPTION: {description}
    AMOUNT : {amount}
    Label: 

ONLY RETURN THE LABEL, NOTHING ELSE

"""

for file in data_list():
    data = pd.read_csv(file)
    for idx, row in data.iterrows():
        messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an excellent data labeler. Your task is to label whatevwer string user sends."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text":prompt.format(description=row['description'],amount=row['amount']},]
        },
    ],
]
