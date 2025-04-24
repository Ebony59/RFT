import torch
import wandb
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from common import *

def apply_ppo_template(
    example,
    tokenizer,
):
    prompt = [{"role": "user", "content": example['question']}]
    
    example['prompt'] = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True)
    example['completion'] = example['answer']
    return example

def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        outputs = tokenizer(
            element['prompt'],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=10,
    )

if __name__ == "__main__":
    BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
    REWARD_MODEL= "ebony59/Qwen2-gsm8k-syn-PRM"
    TRAIN_DATASET = "ebony59/gsm8k-gen-max-distance-sel"
    PROJECT_NAME = "gsm8k_Phi3.5_syn_PPO"
    HF_ID = 'ebony59'
    MODEL_NAME = 'phi3.5-gsm8k-syn-PPO'

    wandb.init(project=PROJECT_NAME, name="syn, 0.5*penalty, r=16, 1e-6 cosine, 1 epoch")

    # Load tokeniser and base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # target_modules='all_liner'
    )
    # model = get_lora_model(model, lora_config)

    ref_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    reward_model = AutoModelForTokenClassification.from_pretrained(
        REWARD_MODEL,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)

    value_model = AutoModelForTokenClassification.from_pretrained(
        REWARD_MODEL,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    dataset = load_dataset(TRAIN_DATASET, split="train")
    dataset = dataset.map(
        apply_ppo_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=list(dataset.features),
        desc="Applying ppo dataset template to train_sft",
    )
    dataset = prepare_dataset(dataset, tokenizer)

    ppo_config = PPOConfig(
        learning_rate=1e-6,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        output_dir="/workspace/ppo_phi3.5_results"
    )

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        peft_config=lora_config
    )

    ppo_trainer.train()
    model.push_to_hub(f'{HF_ID}/{MODEL_NAME}', private=True)
    tokenizer.push_to_hub(f'{HF_ID}/{MODEL_NAME}', private=True)
