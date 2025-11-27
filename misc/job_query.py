import os
import openai
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.FineTuningJob.list(limit=10))
print("==============================================")
latest_job = openai.FineTuningJob.list(limit=10)["data"][0]["id"]
print(openai.FineTuningJob.retrieve(latest_job))
print("==============================================")
print(openai.FineTuningJob.list_events(id=latest_job, limit=1))
