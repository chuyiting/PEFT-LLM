import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig,get_peft_model
from accelerate import Accelerator

from tqdm import tqdm
from collections import defaultdict
import random
from huggingface_hub import login


model_name = 'Qwen/Qwen2.5-Math-7B'
data_path = 'data/full_finetune_data.csv'
cluster_path= 'data/misconception_cluster.csv'
misconception_map_path = 'data/misconception_mapping.csv'
max_length = 256
epochs=5
batch_size=4
lr=5e-5

# Define model
def get_model(model_name, device):
    torch.cuda.empty_cache()
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        bias = 'none'
    )
    
    model = get_peft_model(model, lora_config)

    accelerator = Accelerator(fp16=True)
    model, tokenizer = accelerator.prepare(model, tokenizer)
    model.gradient_checkpointing_enable()
    
    model.to(device)

    return model, tokenizer

def prepare_cluster_dict(cluster_path):
    df = pd.read_csv(cluster_path)
    cluster_dict = defaultdict(list)
    for _, row in df.iterrows():
        cluster_dict[row['Cluster']].append(row['MisconceptionId'])
    return cluster_dict

def prepare_misconception_map(misconception_map_path):
    df = pd.read_csv(misconception_map_path, header=0)
    return dict(zip(df['MisconceptionId'], df['MisconceptionName']))

# Define dataset
class MisconceptionDataset(Dataset):
    def __init__(self, tokenizer, k, data_path, cluster_path, misconception_map_path):
        self.tokenizer = tokenizer
        self.k = k

        self.cluster_dict = prepare_cluster_dict(cluster_path)
        self.misconception_map = prepare_misconception_map(misconception_map_path)
        data = pd.read_csv(data_path, header=0)

        self.question_text = data['QuestionText'] 
        self.construct_name = data['ConstructName']
        self.subject_name = data['SubjectName']
        self.correct_answer_text = data['CorrectAnswerText']
        self.wrong_answer_text = data['WrongAnswerText']
        self.misconception_id = data['MisconceptionId']
        self.misconception_name = data['MisconceptionName']

    def get_negative_examples(self, misconception_id, k, cluster_dict):
        for _, misconceptions in cluster_dict.items():
            if misconception_id in misconceptions:
                # Remove the given misconception_id from the list
                related_misconceptions = [id for id in misconceptions if id != misconception_id]
                
                # If k is smaller than the size of the cluster, return k random misconceptions
                if len(related_misconceptions) >= k:
                    return random.sample(related_misconceptions, k)
                else:
                    # If there are fewer than k misconceptions, return all of them
                    return related_misconceptions
        return []
    
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

        positive = self.misconception_name[idx]
        positive_enc = self.tokenizer(positive, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

        negative_ids = self.get_negative_examples(self.misconception_id[idx], self.k, self.cluster_dict)
        negative = list(map(lambda x: self.misconception_map.get(x, None), negative_ids))
        negative_encs = [self.tokenizer(neg, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt") for neg in negative]

        # Return the tokenized inputs and attention masks
        return {
            'prompt_input_ids': prompt_enc.input_ids.squeeze(0),
            'prompt_attention_mask': prompt_enc.attention_mask.squeeze(0),
            'positive_input_ids': positive_enc.input_ids.squeeze(0),
            'positive_attention_mask': positive_enc.attention_mask.squeeze(0),
            'negative_input_ids': [neg_enc.input_ids.squeeze(0) for neg_enc in negative_encs],
            'negative_attention_mask': [neg_enc.attention_mask.squeeze(0) for neg_enc in negative_encs]
        }

# Multiple Negative Ranking Loss
class MultipleNegativeRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive_embeds, negative_embeds_list):
        """
        Calculates the Multiple Negative Loss (MNRL) using cosine similarity.

        Args:
            anchor (torch.Tensor): Anchor embeddings, shape (batch_size, embed_dim).
            positive_embeds (torch.Tensor): Positive embeddings, shape (batch_size, embed_dim).
            negative_embeds_list (list[torch.Tensor]): List of tensors, 
                each tensor is (batch_size, embed_dim) for negatives.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Normalize embeddings to unit vectors
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive_embeds = F.normalize(positive_embeds, p=2, dim=-1)

        # If there are negatives, stack them into a single tensor
        if len(negative_embeds_list) > 0:
            # Stack negatives along a new dimension (batch_size, num_negatives, embed_dim)
            negative_embeds = torch.stack(negative_embeds_list, dim=1)
            negative_embeds = F.normalize(negative_embeds, p=2, dim=-1)

            # Compute cosine similarity between anchor and negatives
            negative_sim = torch.einsum('bd,bnd->bn', anchor, negative_embeds)  # Shape: (batch_size, num_negatives)
            exp_negatives = torch.exp(negative_sim).sum(dim=-1)  # Shape: (batch_size,)
        else:
            exp_negatives = torch.zeros(anchor.size(0), device=anchor.device)  # No negatives

        # Compute cosine similarity between anchor and positive
        positive_sim = torch.sum(anchor * positive_embeds, dim=-1)  # Shape: (batch_size,)
        exp_positive = torch.exp(positive_sim)  # Shape: (batch_size,)

        # Compute MNRL
        denominator = exp_positive + exp_negatives
        loss = -torch.log(exp_positive / denominator)  # Shape: (batch_size,)

        # Return mean loss
        return loss.mean()

# Finetuning script
def train(model, dataset, device="cuda", epochs=3, batch_size=4, lr=5e-5):

    # Prepare data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = MultipleNegativeRankingLoss()

    model.train()
    for name, param in model.base_model.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}") 
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            prompt_input_ids = batch['prompt_input_ids'].to(device)
            prompt_attention_mask = batch['prompt_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = [neg.to(device) for neg in batch['negative_input_ids']]
            negative_attention_mask = [neg.to(device) for neg in batch['negative_attention_mask']]

            # Forward pass for prompt, positive, and negative examples
            outputs = model(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask)
            prompt_hidden_state = outputs.last_hidden_state[:, -1, :]  # Final hidden state of prompt

            outputs_positive = model(input_ids=positive_input_ids, attention_mask=positive_attention_mask)
            positive_hidden_state = outputs_positive.last_hidden_state[:, -1, :]  # Final hidden state of positive

            negative_hidden_states = []
            for neg_input_id, neg_attention_mask in zip(negative_input_ids, negative_attention_mask):
                outputs_negative = model(input_ids=neg_input_id, attention_mask=neg_attention_mask)
                negative_hidden_states.append(outputs_negative.last_hidden_state[:, -1, :])  # Final hidden state of negative

            # Compute loss
            loss = loss_fn(prompt_hidden_state, positive_hidden_state, negative_hidden_states)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# Example usage
if __name__ == "__main__":
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model(model_name=model_name, device=device)

    # Load dataset
    dataset = MisconceptionDataset(tokenizer, k=25, data_path=data_path, cluster_path=cluster_path, misconception_map_path=misconception_map_path)

    # Train model
    train(model, dataset, device="cuda", epochs=epochs, batch_size=batch_size, lr=lr)

    # Save model
    login(token='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')
    model.push_to_hub("qwen2.5-math-7b-lora-hsft")
    tokenizer.push_to_hub("qwen2.5-math-7b-lora-hsft")
