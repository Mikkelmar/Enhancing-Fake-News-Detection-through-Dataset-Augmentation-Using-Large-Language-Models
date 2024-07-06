import pandas as pd
from datasets import load_dataset


def get_preprocesed_liarData():
  dataset = load_dataset("liar")
  df = pd.DataFrame(dataset['train'])
  columns_to_check = ['statement', 'subject', 'speaker', 'party_affiliation', 'context']

  # Drop rows where any of the specified columns have NaN values
  df = df.dropna(subset=columns_to_check)

  def word_count(statement):
      return len(statement.split())

  # Apply the function and filter the DataFrame
  df = df[df['statement'].apply(word_count) > 8]

  return df
