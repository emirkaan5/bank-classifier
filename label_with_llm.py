import openai
import pandas
import np
from pathlib import Path


DATA_DIR = Path("cleansed_data")
data_list = list(DATA_DIR.rglob("*.csv"))

label_enums = {'1':'grocery','2':'dining','3':'travel','4':'shopping','5':'subscriptions','6':'utilities','7':'health','8':'entertainment','9':'other'}

client = OpenAI()
