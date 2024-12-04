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


from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
import torch

misconception_map_path = os.path.join(os.getcwd(), 'data/misconception_mapping.csv')
top_100_path =  os.path.join(os.getcwd(), "data/top_100_df.csv")


class MisconceptionDataset(Dataset):
    def __init__(self, df, misconception_df, tokenizer, misconception_embeddings):
        self.data = df
        self.misconception_df = misconception_df
        self.tokenizer = tokenizer
        self.misconception_embeddings = misconception_embeddings
    
    def __len__(self):
        return len(self.data)
        
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

    def get_candidates(self, idx):
        candidate_ids = self.data.loc[idx, "top_100"].split(" ")
        return list(map(int, candidate_ids))
            
    def __getitem__(self, idx):
        construct = self.data.loc[idx, 'ConstructName']
        subject = self.data.loc[idx, "SubjectName"]
        question = self.data.loc[idx, "QuestionText"]
        correct= self.data.loc[idx, "CorrectAnswer"]
        wrong = self.data.loc[idx, "IncorrectAnswer"]        
        
        prompt = self.format_prompt(question, construct, subject, correct, wrong)
        prompt_enc = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        candidate_ids = self.get_candidates(idx)
        
        return {
            'prompt_input_ids': prompt_enc.input_ids.squeeze(0),
            'prompt_attention_mask': prompt_enc.attention_mask.squeeze(0),
            'candidate_embeddings': self.misconception_embeddings[candidate_ids, :],
            'candidate_ids': candidate_ids
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

def prepare_misconception_map(misconception_map_path):
    df = pd.read_csv(misconception_map_path, header=0)
    return dict(zip(df['MisconceptionId'], df['MisconceptionName']))


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
    similarity = torch.bmm(misconception_norm, input_norm.unsqueeze(2))  # shape: (B, num_misconceptions, 1)
    
    # Squeeze to remove the last singleton dimension and get shape (B, num_misconceptions)
    similarity = similarity.squeeze(2)  # 
    return similarity



if __name__ == '__main__':

    # unsloth
    max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
    dtype = torch.float16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model_name = 'eddychu/Qwen2-Math-7B-lora-hsft' 

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit, # Offload to CPU or disk while maintaining FP32 precision
    )

    tokenizer = get_chat_template(
                tokenizer,
                chat_template = "alpaca", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
                map_eos_token = True, # Maps <|im_end|> to </s> instead
            )

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    batch_size = 16
    max_length = max_seq_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("data/top_100_df.csv")
    misconception_df = pd.read_csv(misconception_map_path)
    misconception_map = prepare_misconception_map(misconception_map_path)
    misconception_embeddings, misconception_ids = calculate_misconception_hidden_states(model, tokenizer, misconception_map)
    dataset = MisconceptionDataset(df, misconception_df, tokenizer, misconception_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    misconception_map_path = os.path.join(os.getcwd(), 'data/misconception_mapping.csv')
    misconception_map = prepare_misconception_map(misconception_map_path)

    reranked_candidate_ids = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            
            candidate_ids = list(zip(*batch['candidate_ids']))  # Transposes from (N, K) -> (K, N)
            candidate_ids = torch.tensor(candidate_ids, dtype=torch.int64)  # Shape (K, N)

            prompt_input_ids = batch['prompt_input_ids'].to(device)
            prompt_attention_mask = batch['prompt_attention_mask'].to(device)
            
            candidate_embeddings = batch['candidate_embeddings']

            # prompt
            outputs = model(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True)
        
            prompt_last_hidden_state = outputs.hidden_states[-1]
            prompt_last_non_padding_idx = prompt_attention_mask.sum(dim=1) - 1
            prompt_hidden_state = prompt_last_hidden_state[torch.arange(prompt_last_hidden_state.size(0)), prompt_last_non_padding_idx, :].detach().cpu()


            # candidate
            B, _ = prompt_hidden_state.shape 
            similarities = cosine_similarity(prompt_hidden_state, candidate_embeddings)
            print(f'similarity shape: {similarities.shape}')
            sorted_misconception_indices = torch.argsort(similarities, dim=1, descending=True).detach().numpy()
            for i in range(B):
                reranked = candidate_ids[i, sorted_misconception_indices[i]]
                reranked_candidate_ids.append(reranked)
            reranked_candidate_ids.append(reranked)
            print(f'rerank shape: {reranked.shape}')

    reranked_candidate_ids = torch.cat(reranked_candidate_ids, dim=0)
    print(f'final shape: {reranked_candidate_ids.shape}')
            
            

        