from transformers import TrainingArguments, pipeline
from datasets import *
from trl import DPOTrainer
from peft import LoraConfig
import torch
import wandb

from common import *

if __name__ == '__main__':
    BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
    HF_ID = 'ebony59'
    PROJECT_NAME = "gsm8k_Phi3.5_DPO"
    MODEL_NAME = "phi3.5-gsm8k-DPO"
    TRAIN_DATASET = "ebony59/gsm8k-gen-dpo"

    wandb.init(project=PROJECT_NAME, name='1 epoch')

    # Load dataset
    ds = load_dataset(TRAIN_DATASET, split='train')

    df = ds.to_pandas()
    ds = Dataset.from_pandas(df)
    
    print('Length of dataset:',len(ds))

    # Load tokeniser and base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load lora model
    lora_config = LoraConfig(
        lora_alpha=256,
        lora_dropout=0.05,
        r=128,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    
    model = get_lora_model(model, lora_config)

    # Trainer configuration
    training_args = TrainingArguments(
        num_train_epochs=1,
        learning_rate=1e-05,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        do_eval=True,
        per_device_eval_batch_size=1,
        adam_epsilon=1e-08,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.1,
        seed=42,
        logging_steps=10,
        save_steps=1,
        eval_steps=50,
        save_strategy="epoch",
        output_dir=f"data/{MODEL_NAME}",
        hub_model_id="dpo",
        gradient_checkpointing=True,
        bf16=True,
        remove_unused_columns=False,
    )
    
    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=ds,
        tokenizer=tokenizer,
        max_length=1600,
        dataset_num_proc=18,
        peft_config=lora_config,
    )

    dpo_trainer.train()

    # Push to Hub
    model = merge_and_upload_model(model, tokenizer, f'{HF_ID}/{MODEL_NAME}')

    # Trained model verification
    text_generator = pipeline("text-generation", model=model.half(), tokenizer=tokenizer)
    prompt, expected_response = ds['prompt']
    generated_text = text_generator(
            prompt,
            max_new_tokens=500,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id)
    generated_text = generated_text[0]['generated_text'].split('\n<|assistant|>\n')[1]
    print(f'Expected response: {expected_response}')
    print(f'Generated response: {generated_text}')