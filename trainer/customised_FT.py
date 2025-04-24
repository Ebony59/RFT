import torch
import datasets 
import re
import wandb
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer, get_scheduler
from trl import DataCollatorForCompletionOnlyLM
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from peft import LoraConfig, get_peft_model

from trainer.common import *

def apply_tokenizer(
    example,
    tokenizer,
):
    messages = [
      {"role": "user", "content": example['question']},
      {"role": "assistant", "content": example['answer']},
    ]
    
    example["input_ids"] = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False)
    return example

def verify_data_collator(dataset, collator):
    res = []
    printed = False
    for row in dataset:
        _res = collator.torch_call([tokenizer(row['text'])])
        # _res = collator([row])  # collator expects a batch
        pct = (_res['labels'] == -100).float().cpu().numpy().mean()
        res.append(pct)
        if not printed:
            print("collator _res:", {k: v.shape for k, v in _res.items()})
            print("collator pct:", pct)
            print("collator labels:", _res['labels'])
            printed = True
    print("Percentage of examples with 100% masked tokens:", (np.array(res) == 1).mean())

@dataclass
class TrainConfig:
    lr: float = 1e-6
    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation_steps = 4
    device: str = 'cuda'
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    reward_model: AutoModelForTokenClassification,
    reward_tokenizer: AutoTokenizer,
    dataset: datasets.Dataset,
    config: TrainConfig
) -> AutoModelForCausalLM:

    def collate_fn(batch):
        input_ids = [torch.tensor(example['input_ids'], dtype=torch.long) for example in batch]
        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        labels = [ids.clone() for ids in input_ids]

        assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")

        for i, ids in enumerate(input_ids):
            try:
                idx = (ids == assistant_token_id).nonzero(as_tuple=True)[0].item()
                labels[i][:idx + 1] = -100
            except IndexError:
                labels[i][:] = -100

        def left_pad(sequences, padding_value):
            # Flip each sequence
            flipped = [seq.flip(0) for seq in sequences]
            # Pad on the right (which becomes left after flipping back)
            padded = pad_sequence(flipped, batch_first=True, padding_value=padding_value)
            # Flip back to get left-padded sequences
            return padded.flip(1)

        input_ids = left_pad(input_ids, tokenizer.pad_token_id).to(config.device)
        attention_mask = left_pad(attention_mask, 0).to(config.device)
        labels = left_pad(labels, -100).to(config.device)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def loss_fn(batch, global_step):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        base_loss = outputs.loss

        if model.training:
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
    
            logit_strings = tokenizer.batch_decode(preds, skip_special_tokens=True)
            hallucination_penalty = 0.0
            eq_pattern = re.compile(r"<<.*?=.*?>>")
    
            for string in logit_strings:
                equations = re.findall(r"<<([^<>]+)>>", string)
                for eq in equations:
                    try:
                        lhs, rhs = eq.split("=")
                        lhs_processed = re.sub(r'(\d+)(\()', r'\1*\2', lhs.strip())
                        rhs_processed = re.sub(r'(\d+)(\()', r'\1*\2', rhs.strip())
                        
                        lhs_val = eval(lhs_processed.strip())
                        rhs_val = eval(rhs_processed.strip())
                        if abs(lhs_val - rhs_val) > 1e-3:
                            hallucination_penalty += 1.0
                    except Exception:
                        continue

            hallucination_penalty /= max(1, len(logit_strings))

            # Get rewards from PRM model
            prm_inputs = reward_tokenizer(
                logit_strings, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(config.device)
            
            with torch.no_grad():
                token_scores = reward_model(**prm_inputs).logits
                # Get scores for positive class (class 1 = correct reasoning)
                positive_scores = token_scores[:, :, 1]
                # Average scores across tokens
                rewards = (positive_scores * prm_inputs.attention_mask).sum(dim=1) / prm_inputs.attention_mask.sum(dim=1)
            
            adjusted_loss = base_loss + 0.5 * hallucination_penalty - 0.1 * rewards.mean()

            if global_step % 100 == 0:
                print('logit_strings:', logit_strings[0])
                print('rewards:', rewards[0])
        else:
            adjusted_loss = base_loss

        return adjusted_loss

    def get_lora_model(model, lora_config):
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
        return model

    tokenized_dataset = dataset.map(
        apply_tokenizer,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=list(dataset.features),
        desc="Applying tokenizer and chat template",
    )
    
    collate_fn = DataCollatorForCompletionOnlyLM(
        response_template="<|assistant|>\n",
        tokenizer=tokenizer
    )
    verify_data_collator(dataset, collate_fn)

    dataloader = DataLoader(
        tokenized_dataset,
        collate_fn=lambda examples: {
            k: v.to(config.device) 
            for k, v in collate_fn(examples).items()
        },
        # collate_fn=collate_fn,
        batch_size=config.batch_size,
        shuffle=True
    )

    # Load lora model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        # target_modules=["qkv_proj", "o_proj"],
    )
    model = get_lora_model(model, lora_config)

    optimizer = AdamW(model.parameters(), lr=config.lr)

    num_update_steps_per_epoch = len(dataloader) // config.gradient_accumulation_steps
    max_train_steps = config.epochs * num_update_steps_per_epoch
    
    num_warmup_steps = int(max_train_steps * config.warmup_ratio)
    
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    print(f"Scheduler: {config.lr_scheduler_type}, Warmup steps: {num_warmup_steps}, Total steps: {max_train_steps}")

    # This is a pretty bare-bones loop, feel free to add anything particularly useful
    global_step = 0
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            # batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            global_step += 1
            outputs = model(**batch)
            loss = loss_fn(batch, global_step) / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * config.gradient_accumulation_steps

            if (global_step + 1) % 10 == 0:
                base_loss = outputs.loss.item()
                adjusted_loss = loss.item() * config.gradient_accumulation_steps
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log({
                    "base_loss": base_loss,
                    "step_loss": adjusted_loss,
                    "hallucination_penalty": adjusted_loss - base_loss,
                    "learning_rate": current_lr,
                    "global_step": global_step + 1
                })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

    return model

if __name__ == "__main__":
    BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
    REWARD_MODEL= "ebony59/Qwen2-gsm8k-syn-PRM"
    TRAIN_DATASET = "ebony59/gsm8k-gen-max-distance-sel"
    PROJECT_NAME = "gsm8k_Phi3.5_syn_FT"
    HF_ID = 'ebony59'
    MODEL_NAME = 'phi3.5-gsm8k-syn-FT-reward'

    wandb.init(project=PROJECT_NAME, name="syn, 0.5*penalty+0.1*reward, r=16, 1e-6 cosine, 1 epoch")

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
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    reward_model = AutoModelForTokenClassification.from_pretrained(
        REWARD_MODEL,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
    reward_tokenizer.padding_side = "left"
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load dataset
    dataset = datasets.load_dataset(TRAIN_DATASET)['train']
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        desc="Applying chat template to train_sft",
    )
    print('Dataset loaded. Length of dataset:', len(dataset))

    config = TrainConfig()

    #see original model output
    print("original model output")
    validate_output(dataset, model, tokenizer, samples=1)

    trained_model = train(
        model, 
        tokenizer, 
        reward_model,
        reward_tokenizer,
        dataset, 
        config
    )

    trained_model = merge_and_upload_model(trained_model, tokenizer, f'{HF_ID}/{MODEL_NAME}')

    # Trained model verification
    print("new model output")
    validate_output(dataset, trained_model, tokenizer, samples=1)
    



    