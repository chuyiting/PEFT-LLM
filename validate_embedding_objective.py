import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import set_seed, AutoModel, AutoTokenizer
from peft import LoraConfig,get_peft_model
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
import bitsandbytes as bnb

from tqdm import tqdm
from collections import defaultdict
import random
import argparse
import os

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from huggingface_hub import login

import numpy as np

# unsloth
max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.float16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.


model_name = 'unsloth/Qwen2.5-32B-bnb-4bit' #unsloth/
data_path = os.path.join(os.getcwd(), 'data/full_finetune_data.csv')
cluster_path= os.path.join(os.getcwd(), 'data/misconception_cluster.csv')
misconception_map_path = os.path.join(os.getcwd(), 'data/misconception_mapping.csv')
max_length = 256

def apk(actual, predicted, k=25):
    """
    Computes the average precision at k.
    
    This function computes the average prescision at k between two lists of
    items.
    
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """

    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=25):
    """
    Computes the mean average precision at k.
    
    This function computes the mean average prescision at k between two lists
    of lists of items.
    
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    print(f'actual shape {actual.shape}')
    print(f'predicted shape: {predicted.shape}')
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def get_pretrained(model_name, device):
    # this is a hack now, it does handle all the cases for now
    print("use pretrained model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "alpaca", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        map_eos_token = True, # Maps <|im_end|> to </s> instead
    )

    # Freeze all parameters by default
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer

def prepare_misconception_map(misconception_map_path):
    df = pd.read_csv(misconception_map_path, header=0)
    return dict(zip(df['MisconceptionId'], df['MisconceptionName']))

class MisconceptionDataset(Dataset):
    def __init__(self, tokenizer, data_path, misconception_map_path):
        self.tokenizer = tokenizer

        self.misconception_map = prepare_misconception_map(misconception_map_path)
        print(f'data path: {data_path}')
        data = pd.read_csv(data_path, header=0)

        self.question_text = data['QuestionText'] 
        self.construct_name = data['ConstructName']
        self.subject_name = data['SubjectName']
        self.correct_answer_text = data['CorrectAnswerText']
        self.wrong_answer_text = data['WrongAnswerText']
        self.misconception_id = data['MisconceptionId']
        self.misconception_name = data['MisconceptionName']

    
    def format_prompt(self, question_text, construct_name, subject_name, correct_answer_text, wrong_answer_text):

        prompt = f"""Here is a question about {construct_name} ({subject_name}):
        
- Question: {question_text}
- Correct Answer: {correct_answer_text}
- Wrong Answer: {wrong_answer_text}
        
Please identify the likely misconception or reasoning error that led the student to choose the wrong answer. Focus only on explaining the misconception.
"""
    
        message = [
            {"role": "system", "content": "You are a proficient Mathematics teacher. Your goal is to identify the likely misconception or reasoning error that led the student to choose the wrong answer."},
            {"role": "user", "content": prompt.strip()}
        ]
    

        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)

    def __len__(self):
        return len(self.question_text)

    def __getitem__(self, idx):
        question = self.question_text[idx]
        construct = self.construct_name[idx]
        subject = self.subject_name[idx]
        correct = self.correct_answer_text[idx]
        wrong = self.wrong_answer_text[idx]
        
        prompt = self.format_prompt(question, construct, subject, correct, wrong)
        prompt_enc = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

        label = self.misconception_id[idx]

        # Return the tokenized inputs and attention masks
        return {
            'prompt_input_ids': prompt_enc.input_ids.squeeze(0),
            'prompt_attention_mask': prompt_enc.attention_mask.squeeze(0),
            'label': [label]
        }


class MisconceptionNameDataset(Dataset):
    def __init__(self, misconception_map, tokenizer, max_length):
        self.misconception_map = misconception_map
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Prepare list of misconceptions and their ids for fast access
        self.misconceptions = list(misconception_map.items())

    def __len__(self):
        return len(self.misconceptions)

    def __getitem__(self, idx):
        misconception_id, misconception_name = self.misconceptions[idx]
        
        # Tokenize the misconception
        enc = self.tokenizer(misconception_name, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        
        # Return the tokenized data
        return {
            'misconception_id': misconception_id,
            'input_ids': enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0)
        }


# Function to calculate misconception hidden states in batches
def calculate_misconception_hidden_states(model, tokenizer, misconception_map, batch_size=32, max_length=256, device='cuda'):
    misconception_hidden_states = []
    misconception_ids = []

    print("start encoding misconceptions...")
    model.eval()

    # Create a DataLoader for batching
    dataset = MisconceptionNameDataset(misconception_map, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]

            # Get last non-padding token's hidden state
            last_non_padding_idx = attention_mask.sum(dim=1) - 1
            hidden_state = last_hidden_state[torch.arange(last_hidden_state.size(0)), last_non_padding_idx, :]

            # Store the results
            misconception_hidden_states.append(hidden_state.cpu())
            misconception_ids.extend(batch['misconception_id'])

            # Clear cache to avoid memory issues
            del outputs, last_hidden_state, hidden_state
            torch.cuda.empty_cache()

    misconception_hidden_states = torch.cat(misconception_hidden_states, dim=0)  # Concatenate all hidden states
    print("finish encoding misconceptions...")

    return misconception_hidden_states, misconception_ids


def cosine_similarity(input_embeddings, misconception_hidden_states):
    # Normalize both input and misconception embeddings
    input_norm = input_embeddings / input_embeddings.norm(dim=-1, keepdim=True)
    misconception_norm = misconception_hidden_states / misconception_hidden_states.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity using the dot product
    return torch.mm(input_norm, misconception_norm.t())  # shape: (batch_size, num_misconceptions)


def evaluate(model, tokenizer, misconception_map, dataset, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_sorted_misconceptions = []
    all_correct_labels = []

    model.eval()  # Set model to evaluation mode

    for name, param in model.base_model.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}") 

    misconception_embeddings, misconception_ids = calculate_misconception_hidden_states(model, tokenizer, misconception_map)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            prompt_input_ids = batch['prompt_input_ids'].to(device)
            prompt_attention_mask = batch['prompt_attention_mask'].to(device)
            labels = batch['label']
             # Convert labels to the correct shape
            labels = torch.tensor([label[0] for label in labels])  # Extract the label from the list and convert

            # Get the model's output for the batch
            outputs = model(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            
            # Get the last non-padding hidden states for each prompt in the batch
            last_non_padding_idx = prompt_attention_mask.sum(dim=1) - 1
            input_embeddings = last_hidden_state[torch.arange(last_hidden_state.size(0)), last_non_padding_idx, :]

            similarities = cosine_similarity(input_embeddings.cpu(), misconception_embeddings)
            # Sort the misconceptions based on similarity
            sorted_misconception_indices = torch.argsort(similarities, dim=1, descending=True)

            # Map sorted indices to misconception IDs
            sorted_misconception_ids = [[misconception_ids[i] for i in row] for row in sorted_misconception_indices.cpu().numpy()]
            all_sorted_misconceptions.append(sorted_misconception_ids)

            # Ensure labels are a consistent shape
            correct_labels = torch.tensor(labels).unsqueeze(-1).cpu()  # Convert list to tensor and reshape to (B, 1)
            # Append the correctly shaped labels
            all_correct_labels.append(correct_labels.numpy())
        
        all_sorted_misconceptions = np.vstack(all_sorted_misconceptions).astype(int)  # Shape (B, num_misconceptions)
        all_correct_labels = np.vstack(all_correct_labels).astype(int)  # Shape (B, 1)

    return mapk(all_correct_labels, all_correct_labels)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning script with customizable arguments")

    # Add arguments
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    model_name = args.model_name

    # Load model and tokenizer 
    device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
    model, tokenizer = get_pretrained(model_name, device)

    # Load dataset
    dataset = MisconceptionDataset(tokenizer, data_path=data_path, misconception_map_path=misconception_map_path)
    misconception_map = prepare_misconception_map(misconception_map_path)

    # Train model
    map25 = evaluate(model, tokenizer, misconception_map, dataset, batch_size=args.batch_size)
    print(f'MAP@25 score: {map25}')
