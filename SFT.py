from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
from pathlib import Path
from datasets import *
import pandas as pd
import os
import numpy as np
import torch
import wandb

from common import *

TRAINING_SCRIPT_ROOT = Path(__file__).absolute().parent.parent


if __name__ == '__main__':
    BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
    TRAIN_DATASET = "openai/gsm8k"
    PROJECT_NAME = "gsm8k_Phi3.5_SFT"
    HF_ID = 'ebony59'
    MODEL_NAME = 'phi3.5-gsm8k-SFT'

    wandb.init(project=PROJECT_NAME, name="1 epoch")

    # Load tokenizer and base model
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    # Load dataset
    dataset = load_dataset(TRAIN_DATASET, "main")
    print('Dataset loaded. Length of dataset:', len(dataset))

    train_dataset = dataset['train'].map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=list(dataset['train'].features),
        desc="Applying chat template to train_sft",
    )

    test_dataset = dataset['test'].map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=list(dataset['train'].features),
        desc="Applying chat template to train_sft",
    )

    # steps_per_epoch = len(train_dataset)/(per_device_train_batch_size * gradient_accumulation_steps)
    # evaluate reward and alignment scores every half an epoch
    steps_per_epoch = int(len(train_dataset)/16)
    print('steps per epoch:', steps_per_epoch)

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

    # Train model
    training_args = TrainingArguments(
        num_train_epochs=1,
        learning_rate=1e-05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
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
        report_to=['wandb'],
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        # data_collator=collator,
        max_seq_length=1600,
        dataset_text_field="text",
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model()

    # Push to Hub
    trained_model = merge_and_upload_model(model, tokenizer, f'{HF_ID}/{MODEL_NAME}')

    # Trained model verification -> Did it memorize the corpus?
    text_generator = pipeline("text-generation", model=trained_model.half(), tokenizer=tokenizer)
    prompt, expected_response = train_dataset['text'][10].split('\n<|assistant|>\n')
    generated_text = text_generator(
            prompt+'\n<|assistant|>\n',
            max_new_tokens=500,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id)
    generated_text = generated_text[0]['generated_text'].split('\n<|assistant|>\n')[1]
    print(f'Expected response: {expected_response}')
    print(f'Generated response: {generated_text}')