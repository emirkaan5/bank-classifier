# batch_fetch.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import sys
load_dotenv()
# Read API key from environment (do not hardcode secrets)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")

client = OpenAI(api_key=api_key)

batch_id = sys.argv[1]# set your batch id here
result = client.batches.retrieve(batch_id)

# Save response as JSON
with open("batch_result.json", "w", encoding="utf-8") as f:
    json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

