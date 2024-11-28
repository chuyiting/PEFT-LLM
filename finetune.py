from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from unsloth import is_bfloat16_supported

data_path = 'data/finetune_data.csv'

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
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

def get_model(peft=False):
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

    if peft:
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

def get_model_transformers():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct-AWQ")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-32B-Instruct-AWQ")

    return model, tokenizer


# transform the dataset into the 
def get_dataset(data_path, tokenizer):
    
    df = pd.read_csv(data_path, header=0)

    sys_prompt = """You are a math expert. Help users find the misconception in the wrong math answer."""

    question_prompt = """Given a math question, its correct answer and wrong answer, \
    tell me what kind of misconception might the student who answers the wrong answer have. \
    \nHere is an example: Question Text: Which type of graph is represented by the equation \\( y=\\frac{{1}}{{x}} \\)?\n\
    Correct Answer Text: A reciprocal graph\n\
    Wrong Answer Text: A quadratic graph\nThe misconception for the wrong answer is: Confuses reciprocal and quadratic graphs.\n\
    \nNow tell me what misconception does the following wrong answer imply:\n\n"""

    prompts = []
    for question, answer in zip(df['LLM_Response'], df['MisconceptionName']):
        prompt = [
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": question_prompt + question
            },
            {
                "role": "assistant",
                "content": answer
            }

        ]
        prompts.append(prompt)

    dataset = Dataset.from_dict({"chat": prompts})
    dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
    return dataset

# for tokenizer that does not contain default chat template
def get_dataset_alpaca(data_path, tokenizer):
    
    df = pd.read_csv(data_path, header=0)

    instruction = """You are a math expert. Help users find the misconception in the wrong math answer."""

    input = """Given a math question, its correct answer and wrong answer, \
    tell me what kind of misconception might the student who answers the wrong answer have. \
    \nHere is an example: Question Text: Which type of graph is represented by the equation \\( y=\\frac{{1}}{{x}} \\)?\n\
    Correct Answer Text: A reciprocal graph\n\
    Wrong Answer Text: A quadratic graph\nThe misconception for the wrong answer is: Confuses reciprocal and quadratic graphs.\n\
    \nNow tell me what misconception does the following wrong answer imply:\n\n {}"""

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token 
    prompts = []
    for question, answer in zip(df['LLM_Response'], df['MisconceptionName']):
        prompt = alpaca_prompt.format(instruction, input.format(question), answer) + EOS_TOKEN
        prompts.append(prompt)

    dataset = Dataset.from_dict({"formated_text": prompts})
    return dataset


if __name__ == "__main__":
    model, tokenizer = get_model()
    dataset = get_dataset_alpaca(data_path, tokenizer)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "formated_text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 2,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        )
    )

    trainer_stats = trainer.train()

    # model.save_pretrained("lora_model") # Local saving
    # tokenizer.save_pretrained("lora_model") # Local saving

    model.push_to_hub_merged("eddychu/qwen2.5-math-7b-instruct-full-ft", tokenizer, save_method = "merged_16bit", token = "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")
    