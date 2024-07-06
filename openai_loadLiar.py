import pandas as pd
from datasets import load_dataset
import os
from openai_model import query_openai

dataset = load_dataset("liar", trust_remote_code=True)
df = pd.DataFrame(dataset['train'])
columns_to_check = ['statement', 'subject', 'speaker', 'party_affiliation', 'context']

version = "gpt3_v5"
df = df.dropna(subset=columns_to_check)

def word_count(statement):
    return len(statement.split())

# Apply the function and filter the DataFrame
df['word_count'] = df['statement'].apply(word_count)
df = df[df['word_count'] < 8]

def process_samples(dataframe, num_samples=10):
    results = []
    request_ids = []
    response_count = 0
    if os.path.exists(f"processed_ids_{version}.csv"):
        processed_ids = pd.read_csv(f"processed_ids_{version}.csv")['id'].tolist()
    else:
        processed_ids = []

    for index, row in dataframe.iterrows():
        if response_count >= num_samples:
            break
        if row['id'] not in processed_ids:
            promt, response = query_openai(row)
            
            results.append({
                'id': row['id'],
                'label': row['label'],
                'promt': promt,
                'response': response,
                'original_text': row['statement']
            })
            request_ids.append({'id': row['id']})

            # Update the processed IDs file
            pd.DataFrame(results).to_csv(f"openai_responses_{version}.csv", mode='a', index=False, header=not os.path.exists(f"openai_responses_{version}.csv"))
            pd.DataFrame(request_ids).to_csv(f"processed_ids_{version}.csv", mode='a', index=False, header=not os.path.exists(f"processed_ids_{version}.csv"))
            results = []
            request_ids = []
            response_count+=1


    return results

processed_data = process_samples(df,10)