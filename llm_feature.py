import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from bertEncoder import store_encodings
import json
from load_liar import get_preprocesed_liarData
from semantic_bert import generate_text_with_parameters
import logging

# Configure logging to show only warnings and errors
logging.getLogger("transformers").setLevel(logging.ERROR)


def save_text_label_pairs(texts, labels, filename="feature_augmented_text_label_pairs"):
    pairs = [{"text": text, "label": label} for text, label in zip(texts, labels)]
    with open(f'traindata/t5_large/{filename}.json', 'a') as file:
        json.dump(pairs, file, indent=4)
        file.write("\n")

def create_featureaugumentation_text_from(df, model, tokenizer, parameters, instruction):
    texts = df["statement"].tolist()
    labels = df["label"].tolist()
    feature_augmented_texts = []
    labels_pairs = []

    batch_size = 16  # Define batch size
    num_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)
    print("num_batches",num_batches)
    start_batch_index = 0 // batch_size
    print(start_batch_index)
    for batch_index in range(start_batch_index, num_batches):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        batch_texts = texts[start_index:end_index]
        batch_labels = labels[start_index:end_index]

        # Prepare batch of prompts with the instruction
        input_texts = [f"{instruction}\n{text}" for text in batch_texts]
        batch_augmented_texts = generate_text_with_parameters(input_texts, model, tokenizer, parameters)
        batch_augmented_texts = [text.replace(instruction, '').strip() for text in batch_augmented_texts]

        feature_augmented_texts.extend(batch_augmented_texts)
        labels_pairs.extend(batch_labels)

        print(f"Processing batch {batch_index+1}/{num_batches} - Saving progress")
        save_text_label_pairs(batch_augmented_texts, batch_labels, f'best_t5_base_settings_data_{start_index + len(batch_texts) - 1}')

    print("DONE FEATURE AUGMENTATION")



if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Move model to the appropriate device
    

    # Define parameter combinations
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    model = model.to(device)
    model.eval()

    # Define parameter combinations
    
    parameters = {'num_beams': 4, 'early_stopping': True, 'min_length': 128, 'max_length': 200, 'temperature': 0.2, 'top_k': 20, 'top_p': 0.4, 'repetition_penalty': 1.3, 'no_repeat_ngram_size': 2, 'num_return_sequences': 3, 'do_sample': True}
    
    
    instructions = "Extend this narrative maintaining the same tone and style as before:"
    

    # Example text samples
    liar_df = get_preprocesed_liarData()
    #liar_df = liar_df[:1]
    

    # Test the parameters
    create_featureaugumentation_text_from(liar_df, model, tokenizer, parameters, instructions)