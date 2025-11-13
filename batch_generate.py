import json
from openai import OpenAI
import pandas as pd
from pathlib import Path


DATA_DIR = Path("cleansed_data")
data_list = list(DATA_DIR.rglob("*.csv"))

label_enums = {'1':'grocery','2':'dining','3':'travel','4':'shopping','5':'subscriptions','6':'utilities','7':'health','8':'entertainment','9':'other'}

model = "gpt-5-nano-2025-08-07"
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


reqs = []
id = 0
for file in DATA_DIR.iterdir():
    data = pd.read_csv(file)
    for idx, row in data.iterrows():
        payload = {
                "custom_id":f"request-{id}",
                "method":"POST",
                "url":"/v1/chat/completions",
                "body":{
                    "model":model,
                    "messages":[
                        {"role":"system",
                         "content":"You are a an excellent data labeler. Your task is to label whatever string user sends."},
                        {"role":"user","content":prompt.format(description=row['description'],amount=row['amount'])}],
                    "temperature":0.0,"max_tokens":10}
                   }
        print(payload['custom_id'])
        id = id+1
        reqs.append(payload)


with open("batched.jsonl","w",encoding="utf-8") as f_out:
    for obj in reqs:
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")




