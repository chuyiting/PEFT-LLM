import pandas as pd

import torch
import transformers
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
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

from transformers import BitsAndBytesConfig

# unsloth
use_unsloth = False
max_seq_length = 512  # Choose any! We auto support RoPE Scaling internally!
dtype = torch.float16  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model_name = 'unsloth/Qwen2.5-32B-bnb-4bit'  # unsloth/
data_path = os.path.join(os.getcwd(), 'data/full_finetune_data.csv')
cluster_path = os.path.join(os.getcwd(), 'data/misconception_cluster.csv')
misconception_map_path = os.path.join(os.getcwd(), 'data/misconception_mapping.csv')
max_length = 256


# Define model
def get_model(model_name, device, use_lora=True):
    torch.cuda.empty_cache()
    print(f'use model: {model_name}')

    if use_unsloth:
        print('use unsloth')
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template="alpaca",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
            map_eos_token=True,  # Maps <|im_end|> to </s> instead
        )

        if use_lora:
            model = FastLanguageModel.get_peft_model(
                model,
                r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj", ],
                lora_alpha=16,
                lora_dropout=0,  # Supports any, but = 0 is optimized
                bias="none",  # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
                random_state=3407,
                use_rslora=False,  # We support rank stabilized LoRA
                loftq_config=None,  # And LoftQ
                inference_mode=False,
            )
        model.to(torch.float16)
        return model, tokenizer

    #quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    #model = AutoModel.from_pretrained(model_name, trust_remote_code=True, quantization_config=quantization_config)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if use_lora:
        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias='none'#,
            #init_lora_weights="pissa"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
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
        print(f'number of negative examples: {k}')
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
            {"role": "system",
             "content": "You are a proficient Mathematics teacher. Your goal is to identify the likely misconception or reasoning error that led the student to choose the wrong answer."},
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
        positive = self.misconception_name[idx]
        negative_ids = self.get_negative_examples(self.misconception_id[idx], self.k, self.cluster_dict)
        negative = list(map(lambda x: self.misconception_map.get(x, None), negative_ids))

        #prompt_enc = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length,
        #                            return_tensors="pt")


        #positive_enc = self.tokenizer(positive, padding="max_length", truncation=True, max_length=max_length,
        #                              return_tensors="pt")
        #negative_encs = [
        #    self.tokenizer(neg, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt") for
        #    neg in negative]

        # Return the tokenized inputs and attention masks
        return {
            "prompt": prompt,
            "positive": positive,
            "negative": negative
        }


def collate_batch(batches):
    prompts = []
    positives = []
    negatives = []
    for b in batches:
        prompt = b["prompt"]
        positive = b["positive"]
        negative = b["negative"]
        prompts.append(prompt)
        positives.append(positive)
        negatives += negative

    return {
        "prompt": prompts,
        "positive": positives,
        "negative": negatives
    }


# Multiple Negative Ranking Loss
class MultipleNegativeRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, anchor, positive_embeds, negative_embeds):
        """
        Calculates the Multiple Negative Loss (MNRL) using cosine similarity.

        Args:
            anchor (torch.Tensor): Anchor embeddings, shape (batch_size, embed_dim).
            positive_embeds (torch.Tensor): Positive embeddings, shape (batch_size, embed_dim).
            negative_embeds (torch.Tensor): Negative embeddings, shape (batch_size * k, embed_dim)
        Returns:
            torch.Tensor: Scalar loss value.
        """
        k = int(negative_embeds.size(0) / anchor.size(0))
        negative_embeds = negative_embeds.view(-1, k, negative_embeds.size(-1)).permute(1, 0, 2)  # (k, N, E)
        positive_similarity = torch.cosine_similarity(anchor, positive_embeds, dim=1, eps=1e-7)
        negative_similarity = torch.cosine_similarity(anchor, negative_embeds, dim=-1, eps=1e-7).permute(1, 0)
        scores = torch.cat([positive_similarity.unsqueeze(-1), negative_similarity], dim=-1)
        print(scores)

        range_labels = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
        return self.cross_entropy_loss(scores, range_labels)


# Finetuning script
def train(model, tokenizer, dataset, device, loss_fn, epochs=3, batch_size=4, lr=5e-5, max_steps=1000, weight_decay=0.01,
          verbose=False, call_back=None):
    print(f'Number of training epoch: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {lr}')

    for param in model.parameters():
        if param.dtype == torch.qint8:  # Check if it's quantized
            print('clamp quantized')
            param.data = param.data.clamp_(-1.0, 1.0)  # Clamp to a smaller range

    # Prepare data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    optimizer = bnb.optim.AdamW32bit(
        model.parameters(),
        lr=lr,  # Learning rate
        weight_decay=weight_decay  # L2 regularization
    )

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / float(max_steps)))
    # scaler = GradScaler()

    model.train()
    if verbose:
        for name, param in model.base_model.named_parameters():
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    num_steps = 0
    print('start training...')
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        train_pbar = tqdm(dataloader)
        with (torch.autograd.detect_anomaly()):
            for batch in train_pbar:
                optimizer.zero_grad()

                prompt = batch['prompt']
                positive = batch['positive']
                negative = batch['negative']

                prompt_enc = tokenizer(prompt, padding="longest",
                                       truncation=True,
                                       return_tensors="pt",
                                       padding_side="left").to(device)
                positive_enc = tokenizer(positive, padding="longest",
                                         truncation=True,
                                         return_tensors="pt",
                                         padding_side="left").to(device)
                negative_enc = tokenizer(negative, padding="longest",
                                         truncation=True,
                                         return_tensors="pt",
                                         padding_side="left").to(device)

                if max_steps > 0 and num_steps >= max_steps:
                    break

                # Forward pass for prompt, positive, and negative examples
                if not use_unsloth:
                    gen_config = transformers.GenerationConfig(max_length=2048)
                    # anchor
                    outputs = model.generate(**prompt_enc, generation_config=gen_config)
                    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
                    # outputs = model(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask)
                    # prompt_last_non_padding_idx = prompt_attention_mask.sum(dim=1) - 1
                    #prompt_hidden_state = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.size(0)), prompt_last_non_padding_idx, :]

                    # positive
                    outputs_positive = model.generate(**positive_enc, generation_config=gen_config)
                    # outputs_positive = model(input_ids=positive_input_ids, attention_mask=positive_attention_mask)
                    # positive_last_non_padding_idx = positive_attention_mask.sum(dim=1) - 1
                    # positive_hidden_state = outputs_positive.last_hidden_state[torch.arange(outputs_positive.last_hidden_state.size(0)), positive_last_non_padding_idx, :]

                    # negative
                    outputs_negative = model.generate(**negative_enc, generation_config=gen_config)
                    # outputs_negative = model(input_ids=negative_input_ids, attention_mask=negative_attention_mask)
                    # negative_last_non_padding_idx = negative_attention_mask.sum(dim=1) - 1
                    # negative_hidden_states = outputs_negative.last_hidden_state[torch.arange(outputs_negative.last_hidden_state.size(0)), negative_last_non_padding_idx, :]

                else:
                    # with autocast(device_type='cuda'):
                    outputs = model(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask,
                                    output_hidden_states=True)
                    prompt_hidden_state = outputs.hidden_states[-1][:, -1,
                                          :]  # Final hidden state of the last token of prompt

                    outputs_positive = model(input_ids=positive_input_ids, attention_mask=positive_attention_mask,
                                             output_hidden_states=True)
                    positive_hidden_state = outputs_positive.hidden_states[-1][:, -1,
                                            :]  # Final hidden state of the last token of positive misconception

                    negative_hidden_states = []

                    for neg_input_id, neg_attention_mask in zip(negative_input_ids, negative_attention_mask):
                        outputs_negative = model(input_ids=neg_input_id, attention_mask=neg_attention_mask,
                                                 output_hidden_states=True)
                        negative_hidden_states.append(outputs_negative.hidden_states[-1][:, -1,
                                                      :])  # Final hidden state of the last token of negative misconceptions

                loss = loss_fn(prompt_hidden_state, positive_hidden_state, negative_hidden_states)
                loss.backward()
                # scaler.scale(loss).backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                scheduler.step()

                num_steps += 1
                train_pbar.set_description(f"Epoch {epoch}")
                train_pbar.set_postfix({"loss": loss.detach().item()})
                print()

        if call_back is not None:
            print(f'saving model at epoch {epoch}')
            call_back(model, tokenizer)

        if max_steps > 0 and num_steps >= max_steps:
            break
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


def get_loss_function(loss_type):
    """Returns the appropriate loss function based on the `loss_type`."""
    if loss_type == "multiple_negative_ranking":
        print('use multiple negative ranking loss')
        return MultipleNegativeRankingLoss()
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
    parser.add_argument('--loss_type', type=str, default='multiple_negative_ranking',
                        choices=['multiple_negative_ranking', 'triplet', 'info_nce'],
                        help="Type of loss function to use (default: triplet)")
    parser.add_argument('--k', type=int, default=25,
                        help="number of negative examplles")
    parser.add_argument('--max_steps', type=int, default=-1, help="max number of steps for learning rate scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Adam weight decay")
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--use_unsloth', action='store_true')
    parser.add_argument('--hugging_face_repo', type=str)

    args = parser.parse_args()
    use_unsloth = args.use_unsloth
    model_name = args.model_name

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model(model_name=model_name, device=device)

    # Load dataset
    dataset = MisconceptionDataset(tokenizer, k=args.k, data_path=data_path, cluster_path=cluster_path,
                                   misconception_map_path=misconception_map_path)

    # Train model
    train(model, tokenizer, dataset, device=device, loss_fn=get_loss_function(args.loss_type), epochs=args.epoch,
          batch_size=args.batch_size, lr=args.lr, max_steps=args.max_steps, weight_decay=args.weight_decay,
          call_back=save_model)

