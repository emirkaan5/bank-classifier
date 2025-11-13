from openai import OpenAI
from dotenv import load_dotenv
import os
import json
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
stuff =client.files.create(
  file=open("batched.jsonl", "rb"),
  purpose="batch",
  expires_after={
    "anchor": "created_at",
    "seconds": 2592000
  }
)
print(stuff.id)

stuff2 = client.batches.create(
  input_file_id=stuff.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)

print(stuff2)
