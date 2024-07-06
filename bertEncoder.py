import tensorflow as tf
import tensorflow_hub as hub
from keras.api._v2.keras.preprocessing.sequence import pad_sequences
import numpy as np
from transformers import BertTokenizer
import pandas as pd
import os
from PreProcces import load_fakeNews_dataset, load_WELFake_dataset, load_openaiLiar_dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_texts(texts, max_len=128):
  print("ENCODING TEXTS WITH LENGTH:",max_len)
  all_tokens = []
  all_masks = []
  all_segments = []
  
  print_interval = 200  
  
  for index, text in enumerate(texts):
      # Tokenize text and add `[CLS]` and `[SEP]` tokens
      tokens = tokenizer.encode_plus(text, max_length=max_len, truncation=True,
                                      padding='max_length', add_special_tokens=True,
                                      return_tensors='tf')
      # Extract tokens, segments, and masks from the encoded sequence
      all_tokens.append(tokens['input_ids'])
      all_masks.append(tokens['attention_mask'])
      all_segments.append(tokens['token_type_ids'])

      if (index + 1) % print_interval == 0:
          print(f'Encoded {index + 1}/{len(texts)} texts.')
      #print(index, text, tokens['input_ids'],tokens['attention_mask'],tokens['token_type_ids'])
  
  # Stack and reshape for compatibility with BERT input
  all_tokens = tf.reshape(tf.stack(all_tokens), (-1, max_len))
  all_masks = tf.reshape(tf.stack(all_masks), (-1, max_len))
  all_segments = tf.reshape(tf.stack(all_segments), (-1, max_len))
  
  return all_tokens, all_masks, all_segments

def store_encodings(texts, labels, name=""):
    print("encode dataset")
    input_ids, attention_masks, token_type_ids = encode_texts(texts, max_len=128) #128

    np.save(f'embeddings/bertencoder/input_ids{name}.npy', input_ids.numpy())
    np.save(f'embeddings/bertencoder/attention_masks{name}.npy', attention_masks.numpy())
    np.save(f'embeddings/bertencoder/token_type_ids{name}.npy', token_type_ids.numpy())
    np.save(f'embeddings/bertencoder/labels{name}.npy', labels) 

def load_encodings(name=""):
    input_ids = np.load(f'embeddings/bertencoder/input_ids{name}.npy')
    attention_masks = np.load(f'embeddings/bertencoder/attention_masks{name}.npy')
    token_type_ids = np.load(f'embeddings/bertencoder/token_type_ids{name}.npy')
    labels = np.load(f'embeddings/bertencoder/labels{name}.npy')

    return input_ids, attention_masks, token_type_ids, labels
    
if __name__ == '__main__':
    texts, labels = load_openaiLiar_dataset("openai_responses_gpt3_v5_mod")
    store_encodings(texts, labels, "openai_responses_gpt3_v5")