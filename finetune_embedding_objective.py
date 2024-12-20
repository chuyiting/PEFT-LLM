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

# unsloth
use_unsloth=False
max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.float16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.


model_name = 'unsloth/Qwen2.5-Math-7B' #unsloth/
data_path = os.path.join(os.getcwd(), 'data/full_finetune_data.csv')
cluster_path= os.path.join(os.getcwd(), 'data/misconception_cluster.csv')
misconception_map_path = os.path.join(os.getcwd(), 'data/misconception_mapping.csv')
max_length = 256

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

    # Enable gradients only for LoRA parameters
    for name, param in model.named_parameters():
        if "lora" in name.lower():  # Assumes LoRA parameters are named appropriately
            param.requires_grad = True
            print(f"LoRA parameter {name} is set to require gradients")

    return model, tokenizer


# Define model
def get_model(model_name, device, use_lora=True, lora_rank=64, use_pretrained=False):
    
    torch.cuda.empty_cache()
    print(f'use model: {model_name}')
    if use_lora:
        print(f'lora rank: {lora_rank}')

    if use_unsloth:
        print('use unsloth')
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

        if use_lora:
            model = FastLanguageModel.get_peft_model(
                    model,
                    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj",],
                    lora_alpha = 128,
                    lora_dropout = 0.05, # Supports any, but = 0 is optimized
                    bias = "none",    # Supports any, but = "none" is optimized
                    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                    random_state = 3407,
                    use_rslora = False,  # We support rank stabilized LoRA
                    loftq_config = None, # And LoftQ
                    inference_mode = False,
                )
        model.to(torch.float16)
        return model, tokenizer

 
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if use_lora:
        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=lora_rank, 
            lora_alpha=128, 
            lora_dropout=0.05, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
            bias = 'none',
            task_type = "CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)

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
        print (f'number of negative examples: {k}')
        self.k = k

        self.cluster_dict = prepare_cluster_dict(cluster_path)
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

    def get_negative_examples(self, misconception_id, k, cluster_dict):
        related_misconceptions = []
    
        # First, gather all misconceptions related to the misconception_id from the cluster_dict
        for _, misconceptions in cluster_dict.items():
            if misconception_id in misconceptions:
                # Remove the given misconception_id from the list
                related_misconceptions = [id for id in misconceptions if id != misconception_id]
                break

        if len(related_misconceptions) >= k:
            return random.sample(related_misconceptions, k)

         # If we have fewer than k misconceptions, need to add more from other clusters
         
        num_misconception = len(self.misconception_map)
        possible_misconceptions = [i for i in range(num_misconception) if i != misconception_id]
        additional_samples = random.sample(possible_misconceptions, k - len(related_misconceptions))
        related_misconceptions += additional_samples
        return related_misconceptions

    
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


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive_embeds, negative_embeds_list):
        # Compute distances
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive_embeds = F.normalize(positive_embeds, p=2, dim=-1)
        negative_embeds = torch.stack(negative_embeds_list, dim=1)  # (batch_size, num_negatives, embed_dim)
        negative_embeds = F.normalize(negative_embeds, p=2, dim=-1)

        positive_distance = torch.sum((anchor - positive_embeds) ** 2, dim=-1)  # Shape: (batch_size,)
        negative_distance = torch.sum((anchor.unsqueeze(1) - negative_embeds) ** 2, dim=-1)  # Shape: (batch_size, num_negatives)

        # Compute loss
        loss = F.relu(self.margin + positive_distance.unsqueeze(1) - negative_distance)  # Shape: (batch_size, num_negatives)
        return loss.mean()
    
# Multiple Negative Ranking Loss
class MultipleNegativeRankingLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

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
        eps = 1e-7 
        # Normalize embeddings to unit vectors
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive_embeds = F.normalize(positive_embeds, p=2, dim=-1)

        # If there are negatives, stack them into a single tensor
        if len(negative_embeds_list) > 0:
            # Stack negatives along a new dimension (batch_size, num_negatives, embed_dim)
            negative_embeds = torch.stack(negative_embeds_list, dim=1)
            negative_embeds = F.normalize(negative_embeds, p=2, dim=-1)

            # Compute cosine similarity between anchor and negatives
            negative_sim = torch.einsum('bd,bnd->bn', anchor, negative_embeds) / self.temperature  # Shape: (batch_size, num_negatives)
            
            # Clamp similarity values before exponentiation
           #negative_sim = torch.clamp(negative_sim, min=-1.0 + eps, max=1.0 - eps)
            exp_negatives = torch.exp(negative_sim).sum(dim=-1)  # Shape: (batch_size,)
        else:
            exp_negatives = torch.zeros(anchor.size(0), device=anchor.device)  # No negatives

        # Compute cosine similarity between anchor and positive
        positive_sim = torch.sum(anchor * positive_embeds, dim=-1) / self.temperature  # Shape: (batch_size,)

        # Clamp similarity values before exponentiation
        #positive_sim = torch.clamp(positive_sim, min=-1.0 + eps, max=1.0 - eps)
        exp_positive = torch.exp(positive_sim)  # Shape: (batch_size,)

        # Compute MNRL
        denominator = exp_positive + exp_negatives + eps 
        loss = -torch.log(exp_positive / denominator)  # Shape: (batch_size,)

        # Return mean loss
        return loss.mean()

def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=["bias", "LayerNorm.weight"],
        lora_name_list=["lora_A", "lora_B"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
            "lr":
                lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters


# Finetuning script
def train(model, dataset, device, loss_fn, epochs=3, batch_size=4, lr=5e-5, max_steps=1000, weight_decay=0.01, verbose=True, call_back=None, use_scheduler=True, warmup_steps=0):

    print(f'Number of training epoch: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {lr}')

    for param in model.parameters():
        if param.dtype == torch.qint8:  # Check if it's quantized
            print('clamp quantized')
            param.data = param.data.clamp_(-1.0, 1.0)  # Clamp to a smaller range

    # Prepare data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay, lr)
    optimizer = bnb.optim.AdamW8bit(
        optimizer_grouped_parameters,
        lr=lr,         # Learning rate
        weight_decay=weight_decay  # L2 regularization
    )

    if verbose:
        print('params in optimizer')
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                print(param)

    #optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)

    num_iter_per_batch = len(dataset) // batch_size
    total_step = num_iter_per_batch * epochs
    
    if warmup_steps == 0:
        warmup_steps = total_step * 0.03

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # Linear warm-up
        return max(0.0, 1.0 - (step - warmup_steps) / (total_step - warmup_steps))  # Decay

    if use_scheduler:
        print('use learning rate scheduler')
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # scaler = GradScaler(init_scale=1.0, growth_interval=100)

    model.train()
    if verbose:
        for name, param in model.base_model.named_parameters():
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}") 
    
    num_steps = 0
    print('start training...')
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        #with (torch.autograd.detect_anomaly()):
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            
            if max_steps > 0 and num_steps >= max_steps:
                break
            prompt_input_ids = batch['prompt_input_ids'].to(device)
            prompt_attention_mask = batch['prompt_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = [neg.to(device) for neg in batch['negative_input_ids']]
            negative_attention_mask = [neg.to(device) for neg in batch['negative_attention_mask']]

            # Forward pass for prompt, positive, and negative examples
            if not use_unsloth:
                #with autocast(device_type='cuda'):
                outputs = model(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask)
                prompt_hidden_state = outputs.last_hidden_state[:, -1, :]  # Final hidden state of prompt

                outputs_positive = model(input_ids=positive_input_ids, attention_mask=positive_attention_mask)
                positive_hidden_state = outputs_positive.last_hidden_state[:, -1, :]  # Final hidden state of positive

                negative_hidden_states = []
                for neg_input_id, neg_attention_mask in zip(negative_input_ids, negative_attention_mask):
                    outputs_negative = model(input_ids=neg_input_id, attention_mask=neg_attention_mask)
                    negative_hidden_states.append(outputs_negative.last_hidden_state[:, -1, :])  # Final hidden state of negative
            else:
                #with autocast(device_type='cuda'):
                outputs = model(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True)

                prompt_last_hidden_state = outputs.hidden_states[-1]
                prompt_last_non_padding_idx = prompt_attention_mask.sum(dim=1) - 1
                prompt_hidden_state = prompt_last_hidden_state[torch.arange(prompt_last_hidden_state.size(0)), prompt_last_non_padding_idx, :]

                #prompt_hidden_state = outputs.hidden_states[-1][:, -1, :]  # Final hidden state of the last token of prompt

                outputs_positive = model(input_ids=positive_input_ids, attention_mask=positive_attention_mask, output_hidden_states=True)
                positive_last_hidden_state = outputs_positive.hidden_states[-1]
                positive_last_non_padding_idx = positive_attention_mask.sum(dim=1) - 1
                positive_hidden_state = positive_last_hidden_state[torch.arange(positive_last_hidden_state.size(0)), positive_last_non_padding_idx, :]

                #positive_hidden_state = outputs_positive.hidden_states[-1][:, -1, :]  # Final hidden state of the last token of positive misconception

                negative_hidden_states = []
                for neg_input_id, neg_attention_mask in zip(negative_input_ids, negative_attention_mask):
                    outputs_negative = model(input_ids=neg_input_id, attention_mask=neg_attention_mask, output_hidden_states=True)
                    neg_hidden_state = outputs_negative.hidden_states[-1]
                    neg_last_non_padding_idx = neg_attention_mask.sum(dim=1) - 1
                    neg_embedding = neg_hidden_state[torch.arange(neg_hidden_state.size(0)), neg_last_non_padding_idx, :] 
                    negative_hidden_states.append(neg_embedding)
                    #negative_hidden_states.append(outputs_negative.hidden_states[-1][:, -1, :])  # Final hidden state of the last token of negative misconceptions
                
            loss = loss_fn(prompt_hidden_state, positive_hidden_state, negative_hidden_states)
            loss.backward()
            # scaler.scale(loss).backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # scaler.step(optimizer)
            # scaler.update() 
            optimizer.step()
            if use_scheduler:
                scheduler.step()

            num_steps += 1
            print(f"\nLoss: {loss.item()}")

        if call_back is not None:
            print(f'saving model at epoch {epoch}')
            call_back(model, tokenizer)

        if max_steps > 0 and num_steps >= max_steps:
            break
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")
        

def get_loss_function(loss_type, *args):
    """Returns the appropriate loss function based on the `loss_type`."""
    if loss_type == "multiple_negative_ranking":
        print('use multiple negative ranking loss')
        return MultipleNegativeRankingLoss(*args)
    elif loss_type == "triplet":
        print('use triplet loss')
        return TripletLoss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
   
def save_model(model, tokenizer):
    token = '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'
    hugging_face_repo = args.hugging_face_repo
    if not use_unsloth:
        login(token=token)
        model.push_to_hub(hugging_face_repo)
        tokenizer.push_to_hub(hugging_face_repo)
    else:
        model.push_to_hub(hugging_face_repo, token=token)
        tokenizer.push_to_hub(hugging_face_repo, token=token)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning script with customizable arguments")

    # Add arguments
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training (default: 4)")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument('--epoch', type=int, default=4, help="Number of training epochs (default: 4)")
    parser.add_argument('--loss_type', type=str, default='multiple_negative_ranking', choices=['multiple_negative_ranking', 'triplet', 'info_nce'], 
                        help="Type of loss function to use (default: triplet)")
    parser.add_argument('--k', type=int, default=25, 
                        help="number of negative examplles")
    parser.add_argument('--max_steps', type=int, default=-1, help="max number of steps for learning rate scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Adam weight decay")
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--use_unsloth', action='store_true')
    parser.add_argument('--hugging_face_repo', type=str)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--lora_rank', type=int, default=128)
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--turn_off_lrscheduler', action='store_true', default=False)


    args = parser.parse_args()
    use_unsloth = args.use_unsloth
    model_name = args.model_name

    # Load model and tokenizer 
    device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
    if args.use_pretrained:
        model, tokenizer = get_pretrained(model_name, device)
    else:
        model, tokenizer = get_model(model_name=model_name, use_lora=True, lora_rank=args.lora_rank, device=device)


    # Load dataset
    dataset = MisconceptionDataset(tokenizer, k=args.k, data_path=data_path, cluster_path=cluster_path, misconception_map_path=misconception_map_path)

    # Train model
    train(model, dataset, device=device, loss_fn=get_loss_function(args.loss_type, args.temperature), epochs=args.epoch, batch_size= args.batch_size, lr=args.lr, max_steps=args.max_steps, weight_decay=args.weight_decay, use_scheduler = not args.turn_off_lrscheduler, call_back=save_model)
    
