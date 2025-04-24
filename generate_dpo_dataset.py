from datasets import *
import pandas as pd

from validate_data import get_equations, validate_equations
from tqdm import tqdm
from transformers import AutoTokenizer

def gather_dpo_data(prompt, correct_responses, incorrect_responses):
    curr_dpo_ds = []
    for i in range(min(len(correct_responses), len(incorrect_responses))):
        curr_dpo_ds.append({"prompt": prompt, "chosen": correct_responses[i], "rejected": incorrect_responses[i]})
    return curr_dpo_ds

if __name__ == "__main__":
    DATASET = 'ebony59/gsm8k-gen'
    DPO_DATASET = 'ebony59/gsm8k-gen-dpo'
    BASE_MODEL = 'microsoft/Phi-3.5-mini-instruct'

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    dataset = load_dataset(DATASET, split='train')
    ds = dataset.to_pandas().to_dict('records')
    dpo_ds = []

    curr_question = ds[0]['question']
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ds[0]['question']}],
        tokenize=False,
        add_generation_prompt=True
    )
    # orig_response = tokenizer.apply_chat_template(
    #     [{"role": "assistant", "content": ds[0]['orig_answer']}],
    #     tokenize=False,
    #     add_generation_prompt=False
    # )
    orig_response = ds[0]['orig_answer']
    correct_responses = [orig_response]
    incorrect_responses = []

    for d in tqdm(ds):
        if d['question'] != curr_question:
            dpo_ds += gather_dpo_data(prompt, correct_responses, incorrect_responses)

            curr_question = d['question']
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": d['question']}],
                tokenize=False,
                add_generation_prompt=True
            )
            # orig_response = tokenizer.apply_chat_template(
            #     [{"role": "assistant", "content": d['orig_answer']}],
            #     tokenize=False,
            #     add_generation_prompt=False
            # )
            orig_response = d['orig_answer']
            correct_responses = [orig_response]
            incorrect_responses = []

        llm_answer = d['llm_answer']
        # llm_response = tokenizer.apply_chat_template(
        #     [{"role": "assistant", "content": llm_answer}],
        #     tokenize=False,
        #     add_generation_prompt=False
        # )
        llm_response = llm_answer

        if d['correct'] == False:
            incorrect_responses.append(llm_response)
        else:
            equations = get_equations(llm_answer)
            if validate_equations(equations) == True:
                correct_responses.append(llm_response)
            else:
                incorrect_responses.append(llm_response)
    
    dpo_df = pd.DataFrame(dpo_ds)
    dpo_dataset = Dataset.from_pandas(dpo_df)

    print('final dataset length:', len(dpo_dataset))
    print(dpo_dataset[0])
    print(dpo_dataset[100])
    dpo_dataset.push_to_hub(DPO_DATASET, private=True)