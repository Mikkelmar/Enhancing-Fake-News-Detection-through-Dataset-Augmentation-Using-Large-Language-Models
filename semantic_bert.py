# -*- coding: utf-8 -*-

import torch
from transformers import BertModel, BertTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import json
import logging
from load_liar import get_preprocesed_liarData

import statistics
logging.basicConfig(level=logging.WARNING)


bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MODEL_NAME = "t5_large"
VERSION = "0_2"
MODEL_TO_TEST = "google/flan-t5-xl"

def get_mean_embedding(text, tokenizer, model):
    # Tokenize the text in chunks of 512 tokens
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    max_len = 512 - 2  # Account for special tokens [CLS] and [SEP]
    chunks = [token_ids[i: i + max_len] for i in range(0, len(token_ids), max_len)]
    
    embeddings = []
    for chunk in chunks:
        inputs = torch.tensor(chunk).unsqueeze(0)  # Add batch dimension
        outputs = model(inputs)[0].mean(dim=1)  # Get the mean embedding of the output
        embeddings.append(outputs)
    
    # Average the embeddings from all chunks
    embeddings = torch.cat(embeddings, dim=0)
    mean_embedding = embeddings.mean(dim=0, keepdim=True)
    return mean_embedding

def calculate_semantic_similarity(original_text, generated_text):
    with torch.no_grad():
        orig_emb = get_mean_embedding(original_text, bert_tokenizer, bert_model)
        gen_emb = get_mean_embedding(generated_text, bert_tokenizer, bert_model)
        
        cosine_similarity = torch.nn.functional.cosine_similarity(orig_emb, gen_emb)
        return 1 - cosine_similarity.item()


def generate_text_with_parameters(input_texts, model, tokenizer, params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Check and set the pad_token to eos_token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare batch of input texts
    encodings = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Generate text for all inputs at once
    outputs = model.generate(input_ids, attention_mask=attention_mask, **params)

    # Decode all outputs
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return generated_texts

def test_parameter_settings(text_samples, model, tokenizer, params_list, instructions):
    for instruction in instructions:
        for params in params_list:
            print(f"Testing instruction '{instruction}' with parameters: {params}")
            diffs = []
            texts_pairs = []
            for text in text_samples:
                input_text = f"{instruction}\n{text}"
                print(input_text)
                augmented_text = generate_text_with_parameters(input_text, model, tokenizer, params)[0]
                augmented_text = augmented_text.replace(instruction, '').strip()
                diff = calculate_semantic_similarity(text, augmented_text)
                diffs.append(diff)
                texts_pairs.append({
                    "original": text,
                    "augmented_text": augmented_text,
                    "difference": diff
                })

            result = {
                "instruction": instruction,
                "parameters": params,
                "average_difference": np.median(diffs), # median
                "augmented_texts": texts_pairs
            }
            print(f"Storing results from instruction '{instruction}' with parameters: {params}")
            save_results(result)

def save_results(result):
    try:
        with open(f'traindata/{MODEL_NAME}/results_v{VERSION}.json', 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    data.append(result)
    with open(f'traindata/{MODEL_NAME}/results_v{VERSION}.json', 'w') as file:
        json.dump(data, file, indent=4)

def load_results_and_find_best():
    with open(f'traindata/{MODEL_NAME}/results_v{VERSION}.json', 'r') as file:
        data = json.load(file)
    best_result = min(data, key=lambda x: x['average_difference'])
    return best_result

def load_results_and_find_best_median():
    with open(f'traindata/{MODEL_NAME}/results_v{VERSION}.json', 'r') as file:
        data = json.load(file)

    # Update each result entry with the median difference
    for result in data:
        if 'augmented_texts' in result:
            differences = [entry['difference'] for entry in result['augmented_texts']]
            if differences:
                median_difference = statistics.median(differences)
                result['median_difference'] = median_difference
            else:
                result['median_difference'] = float('inf')  # Handle cases with no differences

    # Find the result with the lowest median difference
    best_result = min(data, key=lambda x: x.get('median_difference', float('inf')))

    # Print out the median differences for each parameter set
    for result in data:
        print(f"Parameters: {result['parameters']}")
        print(f"Median Difference: {result.get('median_difference', 'No data')}")

    return best_result

def get_params():
    parameters = [
    # Varied beam numbers with different sampling and penalty settings
        {"num_beams": i, "early_stopping": True, "min_length": 128, "max_length": 200, "temperature": k, "top_k": l, "top_p": m, "repetition_penalty": 1.1, "no_repeat_ngram_size": o, "num_return_sequences": p, "do_sample": True} 
        for i in [4,6,10]  # num_beams from 4 to 12, step 2
        for k in [0.2,0.5,0.8]         # temperature variations
        for l in [20,30,40]         # top_k variations
        for m in [0.4,0.6]         # top_p variations
        for o in [2,4]         # no_repeat_ngram_size variations
        for p in [1,3]        # num_return_sequences variations
    ]
    return parameters
def main():
    # Initialize the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_TEST)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_TO_TEST)
    model.eval()

    # Define parameter combinations

    parameters = [
       {'num_beams': 4, 'early_stopping': True, 'min_length': 128, 'max_length': 200, 'temperature': 0.2, 'top_k': 20, 'top_p': 0.4, 'repetition_penalty': 1.2000000000000002, 'no_repeat_ngram_size': 2, 
'num_return_sequences': 3, 'do_sample': True}
    ]

    instructions = [
        "Add more context and keep writing in the same style:",
        "Continue this text in the original author's style, adding further detail:",
        "Elaborate on this point as if you were the original writer:",
        "Extend this narrative maintaining the same tone and style as before:",
        "Follow up on this with more information, mimicking the original style closely:",
        "Prolong the narrative, keeping the style as close as possible to the original:",
        "Enrich this text by continuing in the same stylistic manner:",
        "Keep the flow going, stick to the original style and deepen the context:",
        "Carry on this conversation in the author’s unique style:",
        "Further develop these ideas in the same writing style:"
    ]

    # Example text samples
    liar_df = get_preprocesed_liarData()
    liar_df = liar_df[:1]
    text_samples = liar_df["statement"]
    

    # Test the parameters
    test_parameter_settings(text_samples, model, tokenizer, parameters, instructions)
    best_setting = load_results_and_find_best()
    print("===================================")
    print(f"Best setting: ")
    print(best_setting["instruction"])
    print(best_setting["average_difference"])
    print(best_setting["parameters"])

if __name__ == "__main__":
    best_setting = load_results_and_find_best()
    print("===================================")
    print(f"Best setting: ")
    print(best_setting["instruction"])
    print(best_setting["average_difference"])
    print(best_setting["parameters"])
