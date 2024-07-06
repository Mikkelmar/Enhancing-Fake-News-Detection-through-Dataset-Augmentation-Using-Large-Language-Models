import os
import json
import numpy as np

from bertEncoder import store_encodings
from load_liar import get_preprocesed_liarData

def load_text_label_pairs_from_json(checkpoints_dir, startswith="feature_augmented_text_label_pair_", max_words=128):
    """
    Load all text-label pairs from JSON files in a specified directory.
    Handles JSON files that may contain multiple JSON documents.
    Assumes files are named with a pattern like 'feature_augmented_text_label_pair_[number].json'.
    """
    # List all JSON files starting with the specified prefix
    json_files = [f for f in os.listdir(checkpoints_dir) if f.startswith(startswith) and f.endswith('.json')]
    # Sort files to ensure correct order based on the number in their filename
    json_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    
    all_texts = []
    all_labels = []
    
    for file in json_files:
        file_path = os.path.join(checkpoints_dir, file)
        with open(file_path, 'r') as file:
            content = file.read()
            try:
                data = json.loads(content)  # Try to load it as a regular JSON array
            except json.JSONDecodeError:
                # If there's a JSONDecodeError, try splitting the content and parsing separately
                parts = content.split('\n')
                data = []
                for part in parts:
                    if part.strip():
                        try:
                            data.extend(json.loads(part))
                        except json.JSONDecodeError:
                            print(f"Error parsing part of the file {file_path}")

            for pair in data:
                words = pair["text"].split()[:max_words]  # Split the text into words and take the first max_words
                truncated_text = " ".join(words)  # Rejoin the words into a string
                all_texts.append(truncated_text)
                all_labels.append(pair["label"])

    
    # Convert labels to a NumPy array for consistency with typical ML workflows
    all_labels = np.array(all_labels)
    label_mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0} #orginal
    #label_mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0}
    # Apply the mapping to transform labels
    transformed_labels = [label_mapping[label] for label in all_labels]
    print(len(all_labels))
    print(f'{transformed_labels.count(1)} samples labels true')
    print(f'{transformed_labels.count(0)} samples labels false')

    return all_texts, transformed_labels

if __name__ == '__main__':
  # Usage example
  

  all_texts, all_labels = load_text_label_pairs_from_json("traindata/t5_base/", "best_t5_base_settings_data_")

  #liar_df = get_preprocesed_liarData()
  #print("LEN",len(liar_df["statement"].tolist()))
  #all_texts.extend(liar_df["statement"].tolist())
  #labels = np.array(liar_df["label"].tolist())
  labels = all_labels
  label_mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0} #orginal
  transformed_labels = [label_mapping[label] for label in labels]
  print(len(transformed_labels))
  print(f'{transformed_labels.count(1)} samples labels true')
  print(f'{transformed_labels.count(0)} samples labels false')

  #all_labels.extend(transformed_labels)
  store_encodings(all_texts, transformed_labels, "best_t5_base_settings_data_v2")