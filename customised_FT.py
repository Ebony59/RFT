import torch
import datasets 
import re
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch.optim import AdamW

from SFT import apply_chat_template

def apply_tokenizer(
    example,
    tokenizer,
):
    messages = [
      {"role": "user", "content": example['question']},
      {"role": "assistant", "content": example['answer']},
    ]
    
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False)
    return example

@dataclass
class TrainConfig:
    lr: float = 3e-5
    epochs: int = 2
    batch_size: int = 4
    device: str = 'cuda'

def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: datasets.Dataset,
    config: TrainConfig
) -> AutoModelForCausalLM:

    tokenized_dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=list(dataset.features),
        desc="Applying tokenizer and chat template",
    )

    def collate_fn(batch):
        return

    def loss_fn(batch):
        # Implement an appropriate loss - note we don't expect this to necessarily
        # be tied to the earlier mentioned paper, just something that is sensible
        return

    optimizer = AdamW(model.parameters(), lr=config.lr)

    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=config.batch_size, shuffle=True)

    # This is a pretty bare-bones loop, feel free to add anything particularly useful
    for epoch in config.epochs:
        model.train()

        for batch in dataloader:
            optimizer.zero_grad()

            loss = loss_fn(**batch)
            loss.backward()

            optimizer.step()

    return model

if __name__ == "__main__":
    BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
    TRAIN_DATASET = "ebony59/gsm8k-gen-sel"
    PROJECT_NAME = "gsm8k_Phi3.5_syn_FT"
    HF_ID = 'ebony59'
    MODEL_NAME = 'phi3.5-gsm8k-syn-FT'

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

    