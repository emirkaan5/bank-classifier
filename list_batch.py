from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import sys
import os
import json
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print(client.batches.list())
print(client.files.retrieve(sys.argv[1]))

with open("retrievedfile.json","w",encoding="utf-8") as f:
    json.dump(client.files.retrieve(sys.argv[1]).model_dump(), f, ensure_ascii=False, indent=2)
