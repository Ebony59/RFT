from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
from peft import get_peft_model
import torch

def apply_chat_template(
    example,
    tokenizer,
):
    messages = [
      {"role": "user", "content": example['question']},
      {"role": "assistant", "content": example['answer']},
    ]
    
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

def print_trainable_parameters(model):
    size = 0
    for name, param in model.named_parameters():
      if param.requires_grad:
          size += param.size().numel()
    print(f'Total number of trainable parameters: {size // 1e6} million')


def get_lora_model(model, lora_config):
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    return model


def merge_and_upload_model(lora_model, tokenizer, model_repo):
    trained_model = lora_model.merge_and_unload()
    tokenizer.push_to_hub(model_repo, private=True)
    trained_model.push_to_hub(model_repo, private=True)
    return trained_model

def get_data_collator(response_template):
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    return collator

def verify_data_collator(dataset, collator, text=None):
    res = []
    for row in dataset:
        _res = collator.torch_call([tokenizer(row['text'])])
        pct = (_res['labels'] == -100).numpy().mean()
        res.append(pct)
    print((np.array(res) == 1).mean())
    if text is not None:
        print(collator.torch_call([tokenizer(text)]))

def validate_output(dataset, model, tokenizer, samples=1):
    for i in range(10, 10+samples):
        with torch.no_grad():
            text_generator = pipeline("text-generation", model=model.half(), tokenizer=tokenizer)
            prompt, expected_response = dataset['text'][i].split('\n<|assistant|>\n')
            generated_text = text_generator(
                    prompt+'\n<|assistant|>\n',
                    max_new_tokens=500,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id)
            generated_text = generated_text[0]['generated_text'].split('\n<|assistant|>\n')[1]
            print(f'Expected response: {expected_response}')
            print(f'Generated response: {generated_text}')