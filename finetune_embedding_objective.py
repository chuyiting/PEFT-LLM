from unsloth import FastLanguageModel

import pandas as pd
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, get_scheduler
from unsloth import is_bfloat16_supported

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm



data_path = 'data/finetune_data.csv'

max_seq_length = 256 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

def get_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        # Can select any from the below:
        # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
        # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
        # And also all Instruct versions and Math. Coding verisons!
        model_name = "unsloth/Qwen2.5-Math-7B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    return model, tokenizer


class AlpacaMathMisconceptionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length):
        """
        Initialize the dataset.

        Args:
        - data_path (str): Path to the CSV file.
        - tokenizer: Tokenizer object for text tokenization.
        - max_seq_length (int): Maximum sequence length for tokenization.
        """
        self.df = pd.read_csv(data_path, header=0)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Define the prompt templates
        self.instruction = (
            "You are a math expert. Help users find the misconception in the wrong math answer."
        )
        self.input_template = """Given a math question, its correct answer and wrong answer, \
tell me what kind of misconception might the student who answers the wrong answer have. \
\nHere is an example: Question Text: Which type of graph is represented by the equation \\( y=\\frac{{1}}{{x}} \\)?\n\
Correct Answer Text: A reciprocal graph\n\
Wrong Answer Text: A quadratic graph\nThe misconception for the wrong answer is: Confuses reciprocal and quadratic graphs.\n\
\nNow tell me what misconception does the following wrong answer imply:\n\n {}"""
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}
"""
        self.eos_token = tokenizer.eos_token or ""  # Use EOS token if available

        # Prepare formatted text prompts
        self.prompts = []
        self.label = []
        for question, answer in zip(self.df["LLM_Response"], self.df["MisconceptionName"]):
            input_text = self.input_template.format(question)
            prompt = self.alpaca_prompt.format(
                self.instruction, input_text
            ) + self.eos_token
            self.prompts.append(prompt)
            self.label.append(answer)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        """
        Return a tokenized version of the formatted prompt at the specified index.

        Args:
        - idx (int): Index of the data sample.

        Returns:
        - dict: Tokenized input with input IDs and attention masks.
        """
        prompt = self.prompts[idx]
        label = self.label[idx]

        # Tokenize the input and labels
        input_encoding = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=self.max_seq_length, return_tensors='pt')
        label_encoding = self.tokenizer(label, padding='max_length', truncation=True, max_length=self.max_seq_length, return_tensors='pt')

        # Extract the token IDs and attention masks
        input_ids = input_encoding['input_ids'].squeeze(0)  # Remove the batch dimension (shape: [max_seq_length])
        attention_mask_input= input_encoding['attention_mask'].squeeze(0)  # Shape: [max_seq_length]
        label_ids = label_encoding['input_ids'].squeeze(0)  # Shape: [max_seq_length]
        attention_mask_label = label_encoding['attention_mask'].squeeze(0) 

        # Flatten the tensors to remove extra dimensions added by return_tensors
        return {
            'input_ids': input_ids,
            'attention_mask_input': attention_mask_input,
            'labels': label_ids,
            'attention_mask_label': attention_mask_label
        }


if __name__ == "__main__":
    model, tokenizer = get_model()
    dataset = AlpacaMathMisconceptionDataset(data_path, tokenizer, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    gradient_accumulation_steps = 4
    global_step = 0

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    # Scheduler
    num_training_steps = 60  # max_steps
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=5,
        num_training_steps=num_training_steps,
    )

    for epoch in range(1):  # Adjust for more epochs if needed
        for batch in tqdm(dataloader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask_input = batch['attention_mask_input'].to(device)
            labels = batch['labels'].to(device)
            attention_mask_label = batch['attention_mask_label'].to(device)


            # Forward pass
            outputs = model(input_ids, attention_mask_input)
            
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps  # Normalize loss

            # Backward pass
            loss.backward()

            if (global_step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step >= num_training_steps:
                break
        if global_step >= num_training_steps:
            break

    print("Training complete.")

    # model.save_pretrained("lora_model") # Local saving
    # tokenizer.save_pretrained("lora_model") # Local saving

    model.push_to_hub_merged("eddychu/Qwen2.5-Math-7B-Instruct-lora-merged", tokenizer, save_method = "merged_16bit", token = "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")
    