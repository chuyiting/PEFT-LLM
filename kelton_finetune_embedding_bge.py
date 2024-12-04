import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from info_nce import InfoNCE, info_nce
# from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import cosine_similarity

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
# import bitsandbytes as bnb

from tqdm import tqdm
from collections import defaultdict
import random
import argparse
import os

data_path = os.path.join(os.getcwd(), 'data/full_finetune_data.csv')
cluster_path = os.path.join(os.getcwd(), 'data/misconception_cluster.csv')
misconception_map_path = os.path.join(os.getcwd(), 'data/misconception_mapping.csv')
max_length = 256

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
    def __init__(self, k, data_path, cluster_path, misconception_map_path):
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
            sample = random.sample(related_misconceptions, k)
            return [self.misconception_map[id] for id in sample]

        # If we have fewer than k misconceptions, need to add more from other clusters

        num_misconception = len(self.misconception_map)
        possible_misconceptions = [i for i in range(num_misconception) if i != misconception_id]
        additional_samples = random.sample(possible_misconceptions, k - len(related_misconceptions))
        related_misconceptions += additional_samples
        return [self.misconception_map[id] for id in related_misconceptions]

    def __len__(self):
        return len(self.question_text)

    def __getitem__(self, idx):
        question = self.question_text[idx]
        construct = self.construct_name[idx]
        subject = self.subject_name[idx]
        correct = self.correct_answer_text[idx]
        wrong = self.wrong_answer_text[idx]

        anchor = " ".join([construct, subject, question, correct])

        positive = self.misconception_name[idx]

        negative_ids = self.get_negative_examples(self.misconception_id[idx], self.k, self.cluster_dict)

        return {
            'anchor': anchor,
            'positive': positive,
            'negatives': negative_ids,
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
        negative_distance = torch.sum((anchor.unsqueeze(1) - negative_embeds) ** 2,
                                      dim=-1)  # Shape: (batch_size, num_negatives)

        # Compute loss
        loss = F.relu(
            self.margin + positive_distance.unsqueeze(1) - negative_distance)  # Shape: (batch_size, num_negatives)
        return loss.mean()

class NewMultipleNegativeRankingLoss(nn.Module):
    def __init__(self, margin=1.0, scale=1.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, anchor, positive_embeds, negative_embeds_list):
        """
        Calculates the Multiple Negative Ranking Loss (MNRL) using cosine similarity.

        Args:
            anchor (torch.Tensor): Anchor embeddings, shape (batch_size, embed_dim).
            positive_embeds (torch.Tensor): Positive embeddings, shape (batch_size, embed_dim).
            negative_embeds_list (list[torch.Tensor]): List of tensors,
                each tensor is (batch_size, embed_dim) for negatives.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Combine positive and negative embeddings
        batch_size = anchor.size(0)
        all_embeddings = [positive_embeds] + negative_embeds_list
        embeddings_b = torch.cat(all_embeddings, dim=0)  # Shape: ((1 + num_negatives) * batch_size, embed_dim)

        # Repeat anchor embeddings for each positive and negative
        embeddings_a = anchor.repeat(1 + len(negative_embeds_list), 1)  # Shape: ((1 + num_negatives) * batch_size, embed_dim)

        # Compute cosine similarity
        scores = torch.einsum("bd,bd->b", embeddings_a, embeddings_b) * self.scale  # Shape: ((1 + num_negatives) * batch_size,)

        # Reshape scores to (batch_size, 1 + num_negatives)
        scores = scores.view(batch_size, 1 + len(negative_embeds_list))  # Shape: (batch_size, 1 + num_negatives)

        # Generate labels: positive samples are at index 0 for each batch
        range_labels = torch.zeros(batch_size, dtype=torch.float16, device=scores.device)  # Shape: (batch_size,)

        # Compute Cross-Entropy Loss
        return self.cross_entropy_loss(scores, range_labels)

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
            negative_sim = torch.einsum('bd,bnd->bn', anchor, negative_embeds)  # Shape: (batch_size, num_negatives)

            # Clamp similarity values before exponentiation
            negative_sim = torch.clamp(negative_sim, min=-1.0 + eps, max=1.0 - eps)
            exp_negatives = torch.exp(negative_sim).sum(dim=-1)  # Shape: (batch_size,)
        else:
            exp_negatives = torch.zeros(anchor.size(0), device=anchor.device)  # No negatives

        # Compute cosine similarity between anchor and positive
        positive_sim = torch.sum(anchor * positive_embeds, dim=-1)  # Shape: (batch_size,)

        # Clamp similarity values before exponentiation
        positive_sim = torch.clamp(positive_sim, min=-1.0 + eps, max=1.0 - eps)
        exp_positive = torch.exp(positive_sim)  # Shape: (batch_size,)

        # Compute MNRL
        denominator = exp_positive + exp_negatives + eps
        loss = -torch.log(exp_positive / denominator)  # Shape: (batch_size,)

        # Return mean loss
        return loss.mean()
    
def generate_new_positive(anchor_emb, positive_emb, hardest_negative_emb):
    anchor_embedding_norm = F.normalize(anchor_emb, p=2, dim=1)
    positive_embedding_norm = F.normalize(positive_emb, p=2, dim=1)
    negative_embedding_norm = F.normalize(hardest_negative_emb, p=2, dim=1)
    d1 = torch.sum(anchor_embedding_norm * positive_embedding_norm, dim=1, keepdim=True) 
    d2 = torch.sum(anchor_embedding_norm * negative_embedding_norm, dim=1, keepdim=True) 

    w1 = d2 / (d1 + d2 + 1e-8)  
    w2 = d1 / (d1 + d2 + 1e-8)  

    new_positive = w1 * positive_emb + w2 * hardest_negative_emb  
    return new_positive

def generate_new_negatives(hard_negatives, s):
    batch_size, num_negatives, embedding_dim = hard_negatives.shape
    new_negatives = []

    for b in range(batch_size):
        batch_negatives = hard_negatives[b]
        batch_new_negatives = []

        for _ in range(s):
            i, j = torch.randperm(num_negatives)[:2]
            alpha = torch.rand(1).item() 
            mixed_negative = alpha * batch_negatives[i] + (1 - alpha) * batch_negatives[j]
            batch_new_negatives.append(mixed_negative)

        batch_new_negatives = torch.stack(batch_new_negatives) 

        combined_negatives = torch.cat((batch_negatives, batch_new_negatives), dim=0)
        new_negatives.append(combined_negatives)

    return torch.stack(new_negatives)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Finetuning script
def train(model, tokenizer, dataset, device, new_negative_num, loss_fn, epochs=3, batch_size=4, lr=5e-5, max_steps=1000, weight_decay=0.01,
          verbose=False):
    print(f'Number of training epoch: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {lr}')

    # Prepare data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,  # Learning rate
        weight_decay=weight_decay
    )

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / float(max_steps)))
    # scaler = GradScaler()

    model.train()
    if verbose:
        for name, param in model.base_model.named_parameters():
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    num_steps = 0
    print('start training...')
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        train_pbar = tqdm(dataloader)
        with torch.autograd.detect_anomaly():
            for batch in train_pbar:
                optimizer.zero_grad()

                if max_steps > 0 and num_steps >= max_steps:
                    break

                anchor = batch['anchor']
                positive = batch['positive']
                negatives = batch['negatives']

                print(f"neg shape: {len(negatives)}, {len(negatives[0])}")

               # Tokenize input
                anchor_inputs = tokenizer(anchor, padding=True, truncation=True, return_tensors="pt").to(device)
                positive_inputs = tokenizer(positive, padding=True, truncation=True, return_tensors="pt").to(device)
                negative_inputs = [tokenizer(neg, padding=True, truncation=True, return_tensors="pt").to(device) for neg in negatives]

                # Forward pass
                anchor_embeddings = mean_pooling(model(**anchor_inputs), anchor_inputs['attention_mask'])
                positive_embeddings = mean_pooling(model(**positive_inputs), positive_inputs['attention_mask'])
                negative_embeddings = torch.stack([
                    mean_pooling(model(**neg_input), neg_input['attention_mask']) for neg_input in negative_inputs
                ])
                negative_embeddings = negative_embeddings.permute(1, 0, 2)
                print(f"anchor: {anchor_embeddings.shape}, pos: {positive_embeddings.shape}, neg: {negative_embeddings.shape}")

                new_negative_embeddings = generate_new_negatives(negative_embeddings, new_negative_num)
                print(f'new neg embeddings: {new_negative_embeddings.shape}, first neg: {negative_embeddings[:, 0, :].shape}')

                new_positive_embeddings = generate_new_positive(anchor_embeddings, positive_embeddings, negative_embeddings[:, 0, :])
                print(f'new positive shape: {new_positive_embeddings.shape}')

                # anchor_embeddings.requires_grad_()
                # positive_embeddings.requires_grad_()
                # negative_embeddings.requires_grad_()
                # new_positive_embeddings.requires_grad_()
                # new_negative_embeddings.requires_grad_()

                origin_loss = loss_fn(anchor_embeddings, positive_embeddings, new_negative_embeddings)
                synthetic_loss = loss_fn(anchor_embeddings, new_positive_embeddings, negative_embeddings)
                # loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                loss = origin_loss + synthetic_loss
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
                print(f"Epoch {epoch} loss: {loss.detach().item() / 2}")

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
    parser.add_argument('--model_name', type=str, default='deberta')
    parser.add_argument('--new_negative', type=int, default=5)

    args = parser.parse_args()
    model_name = args.model_name

    if model_name == "deberta":
        # model = SentenceTransformer('embedding-data/deberta-sentence-transformer')
        tokenizer = AutoTokenizer.from_pretrained('embedding-data/deberta-sentence-transformer')
        model = AutoModel.from_pretrained('embedding-data/deberta-sentence-transformer')
    else:
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load dataset
    dataset = MisconceptionDataset(k=args.k, data_path=data_path, cluster_path=cluster_path,
                                   misconception_map_path=misconception_map_path)

    # Train model
    train(model, tokenizer, dataset, device=device, new_negative_num=args.new_negative, loss_fn=InfoNCE(negative_mode='paired'), epochs=args.epoch,
          batch_size=args.batch_size, lr=args.lr, max_steps=args.max_steps, weight_decay=args.weight_decay)