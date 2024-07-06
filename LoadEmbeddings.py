import numpy as np
import os

types = {
    "bert_en_uncased": "bert_embeddings2_checkpoint_",
    "bert": "bert_embeddings_checkpoint_",
    "bert_128": "bert_embeddings_maxL128_checkpoint_",
    }

def load_all_embeddings_from_checkpoints(checkpoints_dir, model_type="bert"):
    # List all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith(types[model_type]) and f.endswith('.npz')]
    # Sort files to ensure correct order
    checkpoint_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    
    all_embeddings = []
    all_labels = []
    all_texts = []

    for file in checkpoint_files:
        
        file_path = os.path.join(checkpoints_dir, file)
        with np.load(file_path) as data:
            embeddings = data['embeddings']
            labels = data['labels']
            if model_type == "bert":
                texts = data['text'] 
            else:
                texts = data['texts'] 

            all_texts.append(texts)
            all_embeddings.append(embeddings)
            all_labels.append(labels)
    
    # Concatenate all embeddings and labels
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)
    all_texts = np.concatenate(all_texts)
    
    return all_embeddings, all_labels, all_texts

# Example usage
if __name__ == '__main__':
    checkpoints_dir = './embeddings/'
    all_embeddings, all_labels, all_texts = load_all_embeddings_from_checkpoints(checkpoints_dir)
    print(f"Loaded embeddings shape: {all_embeddings.shape}")
    print(f"Loaded labels shape: {all_labels.shape}")
    print(f"Loaded texts shape: {all_texts.shape}")
    for i in range(10):
        print(all_texts[i][:13])
