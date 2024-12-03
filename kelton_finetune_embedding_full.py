import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from info_nce import InfoNCE, info_nce
from sklearn.metrics.pairwise import cosine_similarity

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
    def __init__(self, k, data_path, cluster_path, misconception_map_path, model, device):
        print(f'number of negative examples: {k}')
        self.k = k
        self.model = model
        self.device = device

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
        self.precompute_negatives()

    def precompute_negatives(self):
        self.model.eval()
        misconception_embeddings = self.model.encode(list(self.misconception_map.values()), device=self.device, normalize_embeddings=True)
        
        anchors = [" ".join([c, s, q, a, w]) for c, s, q, a, w in zip(
            self.construct_name, self.subject_name, self.question_text, self.correct_answer_text, self.wrong_answer_text)]
        anchor_embeddings = self.model.encode(anchors, device=self.device, normalize_embeddings=True)

        similarity_matrix = cosine_similarity(anchor_embeddings, misconception_embeddings)
        top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :self.k]

        self.precomputed_negatives = [[self.misconception_map[idx] for idx in row] for row in top_k_indices]
        print("Negative examples precomputed for this epoch.")

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

        anchor = " ".join([construct, subject, question, correct, wrong])

        positive = self.misconception_name[idx]

        # negative_ids = self.get_negative_examples(self.misconception_id[idx], self.k, self.cluster_dict)
        negative_ids = self.precomputed_negatives[idx]

        return {
            'anchor': anchor,
            'positive': positive,
            'negatives': negative_ids,
        }
    
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

def generate_new_negatives(hard_negatives, new_negative_num):
    batch_size, num_negatives, embedding_dim = hard_negatives.shape
    new_negatives = []

    for b in range(batch_size):
        batch_negatives = hard_negatives[b]
        batch_new_negatives = []

        for _ in range(new_negative_num):
            i, j = torch.randperm(num_negatives)[:2]
            alpha = torch.rand(1).item() 
            mixed_negative = alpha * batch_negatives[i] + (1 - alpha) * batch_negatives[j]
            batch_new_negatives.append(mixed_negative)

        batch_new_negatives = torch.stack(batch_new_negatives) 

        combined_negatives = torch.cat((batch_negatives, batch_new_negatives), dim=0)
        new_negatives.append(combined_negatives)

    return torch.stack(new_negatives)

# Finetuning script
def train(model, k, device, new_negative_num, synthetic_weight, loss_fn, epochs=4, batch_size=8, lr=3e-5, max_steps=1000, weight_decay=0.01,
          model_name='deberta', verbose=False):
    print(f'Number of training epoch: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {lr}')

    # Prepare data
    optimizer = AdamW(
        model.parameters(),
        lr=lr,  # Learning rate
        weight_decay=weight_decay
    )

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / float(max_steps)))
    # scaler = GradScaler()

    if verbose:
        for name, param in model.base_model.named_parameters():
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    num_steps = 0
    print('start training...')
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Prepare data in each epoch
        dataset = MisconceptionDataset(k=k, data_path=data_path, cluster_path=cluster_path,
                                   misconception_map_path=misconception_map_path, model=model, device=device)
        model.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

                anchor_embeddings = torch.tensor(model.encode(anchor, normalize_embeddings=True, device=device), device=device)
                positive_embeddings = torch.tensor(model.encode(positive, normalize_embeddings=True, device=device), device=device)

                negative_embeddings = torch.tensor([model.encode(negative_row, normalize_embeddings=True, device=device) for negative_row in negatives], device=device)
                negative_embeddings = negative_embeddings.permute(1, 0, 2)

                new_negative_embeddings = generate_new_negatives(negative_embeddings, new_negative_num)
                new_positive_embeddings = generate_new_positive(anchor_embeddings, positive_embeddings, negative_embeddings[:, 0, :])

                anchor_embeddings.requires_grad_()
                positive_embeddings.requires_grad_()
                new_positive_embeddings.requires_grad_()
                new_negative_embeddings.requires_grad_()

                origin_loss = loss_fn(anchor_embeddings, positive_embeddings, new_negative_embeddings)
                synthetic_loss = loss_fn(anchor_embeddings, new_positive_embeddings, new_negative_embeddings)
                loss = origin_loss + (synthetic_weight * synthetic_loss)
                loss.backward()
                # scaler.scale(loss).backward()

                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                scheduler.step()

                num_steps += 1
                train_pbar.set_description(f"Epoch {epoch}")
                train_pbar.set_postfix({"loss": origin_loss.detach().item()})
                print(f"Epoch {epoch} loss: {origin_loss.detach().item()}")

                total_loss += origin_loss.detach().item()

        if max_steps > 0 and num_steps >= max_steps:
            break
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")
    
    model_path = "models/" + model_name
    model.save_pretrained(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning script with customizable arguments")

    # Add arguments
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training (default: 4)")
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument('--epoch', type=int, default=4, help="Number of training epochs (default: 4)")
    parser.add_argument('--k', type=int, default=25,
                        help="number of negative examplles")
    parser.add_argument('--max_steps', type=int, default=-1, help="max number of steps for learning rate scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Adam weight decay")
    parser.add_argument('--model_name', type=str, default='deberta')
    parser.add_argument('--new_negative', type=int, default=5)
    parser.add_argument('--synthetic_weight', type=float, default=0.5)

    args = parser.parse_args()
    model_name = args.model_name

    if model_name == "deberta":
        model = SentenceTransformer('embedding-data/deberta-sentence-transformer')
    else:
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    train(model, k=args.k, device=device, new_negative_num=args.new_negative, synthetic_weight=args.synthetic_weight, loss_fn=InfoNCE(negative_mode='paired'), epochs=args.epoch,
          batch_size=args.batch_size, lr=args.lr, max_steps=args.max_steps, weight_decay=args.weight_decay, model_name=model_name)