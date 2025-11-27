import os
import openai
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")

file = 'prompts/finetune_data_new.jsonl'
file_id = (openai.File.create(
  file=open(file, "rb"),
  purpose='fine-tune'
))["id"]

# subprocess.run('openai api files.create   -f prompts/finetune_data_new.jsonl -p fine-tune')
# pd.read_json(path_or_buf=file, lines=True)
# print(openai.FineTuningJob.create(training_file=file_id,
# 				model="gpt-3.5-turbo", suffix='GenSimNew'))