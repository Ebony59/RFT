import torch
import datasets 
import re
import wandb
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model

from SFT import merge_and_upload_model, print_trainable_parameters

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
        _res = collator([row])  # collator expects a batch
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
    lr: float = 1e-5
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps = 16
    device: str = 'cuda'

def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
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

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(config.device)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0).to(config.device)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(config.device)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def loss_fn(batch):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        base_loss = output.loss

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

        hallucination_penalty /= len(logit_strings)
        adjusted_loss = base_loss + 0.5 * halllucination_penalty
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

    # data_collator = DataCollatorForCompletionOnlyLM(
    #     response_template="<|assistant|>\n",
    #     tokenizer=tokenizer
    # )

    verify_data_collator(tokenized_dataset, collate_fn)

    dataloader = DataLoader(
        tokenized_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        shuffle=True
    )

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

    optimizer = AdamW(model.parameters(), lr=config.lr)

    # This is a pretty bare-bones loop, feel free to add anything particularly useful
    global_step = 0
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            global_step += 1
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * config.gradient_accumulation_steps

            if (global_step + 1) % 10 == 0:
                wandb.log({"step_loss": loss.item() * config.gradient_accumulation_steps, "global_step": global_step + 1})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

    return model

if __name__ == "__main__":
    BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
    TRAIN_DATASET = "openai/gsm8k"
    PROJECT_NAME = "gsm8k_Phi3.5_syn_FT"
    HF_ID = 'ebony59'
    MODEL_NAME = 'phi3.5-gsm8k-syn-FT'

    wandb.init(project=PROJECT_NAME, name="1 epoch")

    # Load tokeniser and base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Load dataset
    dataset = datasets.load_dataset(TRAIN_DATASET, "main")['train']
    print('Dataset loaded. Length of dataset:', len(dataset))

    config = TrainConfig()
    trained_model = train(model, tokenizer, dataset, config)

    trained_model = merge_and_upload_model(trained_model, tokenizer, f'{HF_ID}/{MODEL_NAME}')

    # Trained model verification
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



    