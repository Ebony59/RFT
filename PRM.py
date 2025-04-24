import wandb

from datasets import load_dataset
from trl import PRMConfig, PRMTrainer
from peft import LoraConfig
from transformers import AutoModelForTokenClassification, AutoModelForCausalLM, AutoTokenizer

from common import *

if __name__ == "__main__":
    BASE_MODEL = "Qwen/Qwen2-0.5B"
    TRAIN_DATASET = "ebony59/gsm8k-gen-stepwise-1label"
    PROJECT_NAME = "gsm8k_Qwen2_syn_PRM"
    HF_ID = 'ebony59'
    MODEL_NAME = 'Qwen2-gsm8k-syn-PRM-1label'

    wandb.init(project=PROJECT_NAME, name="1 epoch, 1 label")

    # Load tokeniser and base model
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(TRAIN_DATASET, split='train')
    print(f"Dataset loaded with {len(train_dataset)} examples")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="TOKEN_CLS",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'classifier'],
        modules_to_save=None
    )

    # model = get_lora_model(model, lora_config)

    training_args = PRMConfig(
        output_dir='/workspace/PRM_output',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb"
    )

    trainer = PRMTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        # peft_config=lora_config
    )

    print("Starting PRM training...")
    trainer.train()

    print(model)
    
    tokenizer.push_to_hub(f'{HF_ID}/{MODEL_NAME}', private=True)
    model.push_to_hub(f'{HF_ID}/{MODEL_NAME}', private=True)
    