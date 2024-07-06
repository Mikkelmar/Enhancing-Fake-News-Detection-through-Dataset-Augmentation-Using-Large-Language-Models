import tensorflow_hub as hub
import pandas as pd
import tensorflow as tf
import numpy as np

def store_texts_to_bert_embeddings(csv_file_path):
    data = pd.read_csv(csv_file_path)
    
    # Drop rows with any NA values in 'title', 'text', or 'label' columns
    data = data.dropna(subset=['title', 'text', 'label'])
    
    # Merge 'title' and 'text' columns
    data['merged_text'] = data['title'] + " " + data['text']
    
    texts = data['merged_text'].tolist()
    labels = data['label'].to_numpy()

    # Load the preprocessing model from TensorFlow Hub
    preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    preprocess_model = hub.KerasLayer(preprocess_url)

    # Load the BERT model from TensorFlow Hub
    bert_model_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2"
    bert_model = hub.KerasLayer(bert_model_url, trainable=True)

    embeddings_list = []
    labels_list = []
    texts_list = []
    checkpoint_size = 300  # Save progress every n embeddings
    start_index = 0

    for i, text in enumerate(texts[start_index:], start=start_index):
        # Preprocess the text
        preprocessed_text = preprocess_model([text])
        # Extract embeddings
        with tf.GradientTape() as tape:
            embedding = bert_model(preprocessed_text)['pooled_output']
            
        embeddings_list.append(embedding.numpy())
        labels_list.append(labels[i])
        texts_list.append(text)
        
        if i % 25 == 0:
            print(f'{i}/{len(texts)} done...')

        # Checkpoint: Save progress every 500 embeddings
        if (i % checkpoint_size == 0 and i != start_index) or (i + 1) == len(texts):
            current_embeddings = np.vstack(embeddings_list)
            current_labels = labels_list
            current_texts = texts_list
            checkpoint_file = f'embeddings/bert_embeddings_uncased_checkpoint_{i}.npz'
            np.savez(checkpoint_file, embeddings=current_embeddings, labels=current_labels, texts=current_texts)
            print(f"Checkpoint saved: {checkpoint_file}")
            embeddings_list = []
            labels_list = []
            texts_list = []

    print("Final embeddings and labels saved to file.")