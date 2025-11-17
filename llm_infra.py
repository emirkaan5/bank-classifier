from transformers import pipeline
import torch
import pandas as pd
pipe = pipeline("text-generation", model="google/gemma-3-4b-it", device="mps", torch_dtype=torch.bfloat16)
import re
from pathlib import Path

DATA_DIR = Path("cleansed_data")
data_list = list(DATA_DIR.rglob("*.csv"))
OUT_DIR = Path("llm_gen/gemma")
OUT_DIR.mkdir(exist_ok=True)

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

for file in data_list:
    data = pd.read_csv(file)
    print(f"data is being labeled: {file}")
    for idx, row in data.iterrows():
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a an excellent data labeler. Your task is to label whatever string user sends."},]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt.format(description=row['description'],amount=row['amount'])},]
                },
            ],
        ]

        output = pipe(messages, max_new_tokens=50)
        out = output[0][0]["generated_text"][2]["content"]
        # print(out)
        data.loc[idx, "label"] = re.sub(r'[^\w\s]','',out.lower())
        save_path = OUT_DIR / ("gemma_4b_" + file.name)
        data.to_csv(save_path, index=False)

"""
    [
        {'generated_text': 
            [
                {'role': 'system', 
                'content': 
                    [
                        {'type': 'text', 
                        'text': 'You are a an excellent data labeler. Your task is to label whatever string user sends.'}
                    ]
                }, 
                {'role': 'user', 
                'content': 
                    [   
                        {'type': 'text', 
                        'text': '\nFollowing text, with an amount of expense is from a bank statement document.\nYou are tasked to label this text into following 9 categories:\n    1.Grocery\n    2.Dining\n    3.Travel\n    4.Shopping\n    5.Subscription\n    6.Utilites\n    7.Health\n    8.Entertainment\n    9.Other\n\nIf you are absolutely uncertain on which one to put one label in, put to 9.\nYou must always give a guess. Here is an example:\n\n    DESCRIPTION: HANGAR PUB WINGS AMHER10 UNIVERSITY DR AMHERST 01002 MA USA 1%\n    AMOUNT: \t50.44\n    Label: Dining\n\nHere is data to be labeled: \n\n    DESCRIPTION: Previous Monthly Balance\n    AMOUNT : 0.0\n    Label: \n\nONLY RETURN THE LABEL, NOTHING ELSE\n\n'
                        }
                    ]   
                }, 
                {'role': 'assistant', 
                'content': 'Label: Utilities'
                }
            ]
        }
    ]
"""
