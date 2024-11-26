from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd

data_path = 'data/finetune_data.csv'

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

def get_model():
    # Load the Qwen model
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # or load_in_4bit=True for even lower memory
        device_map="auto",  # Automatically map model to available GPUs
        trust_remote_code=True
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank updates
        lora_alpha=16,  # Scaling factor
        target_modules=["query_key_value"],  # Modules to apply LoRA
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",  # Bias setting
        task_type="CAUSAL_LM"  # Task type
    )

    # Add LoRA adapters to the model
    model = get_peft_model(model, lora_config)
    return tokenizer, model

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

    # Training arguments
    training_args = TrainingArguments(
        push_to_hub=True,
        output_dir="./lora-finetuned-qwen",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_dir="./logs",
        save_steps=200,
        save_total_limit=2,
        fp16=True,  # Enable mixed precision
        report_to="none"
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset
    )

    # Start training
    trainer.train()
    trainer.push_to_hub("eddychu/Qwen2.5-Math-7B-Instruct-lora-merged")
    