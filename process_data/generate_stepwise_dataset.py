from datasets import *
import pandas as pd
import Levenshtein

from validate_data import get_equations, validate_equations
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    DATASET = 'ebony59/gsm8k-gen'
    STEPWISE_DATASET = 'ebony59/gsm8k-gen-stepwise'
    BASE_MODEL = 'microsoft/Phi-3.5-mini-instruct'

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    dataset = load_dataset(DATASET, split='train')
    ds = dataset.to_pandas().to_dict('records')
    stepwise_ds = []

    curr_question = ds[0]['question']
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ds[0]['question']}],
        tokenize=False,
        add_generation_prompt=True
    )
    orig_equations = '|'.join(get_equations(ds[0]['orig_answer']))
    orig_response = ds[0]['orig_answer'].split('\n')
    stepwise_ds.append({
        'prompt': prompt,
        'completions': orig_response,
        'labels': [True for _ in range(len(orig_response))]
    })

    for d in tqdm(ds):
        if d['question'] != curr_question:
            curr_question = d['question']
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": d['question']}],
                tokenize=False,
                add_generation_prompt=True
            )
            orig_equations = '|'.join(get_equations(d['orig_answer']))
            orig_response = d['orig_answer'].split('\n')
            stepwise_ds.append({
                'prompt': prompt,
                'completions': orig_response,
                'labels': [True for _ in range(len(orig_response))]
            })

        
        equations = '|'.join(get_equations(d['llm_answer']))

        if Levenshtein.distance(equations, orig_equations) == 0:
            continue

        llm_response = d['llm_answer'].split('\n')
        has_error = False
        labels = []

        for response in llm_response:
            equations = get_equations(response)
            if validate_equations(equations) == False:
                has_error = True
            if has_error:
                labels.append(False)
            else:
                labels.append(True)

        if not has_error and not d['correct']:
            continue
            
        stepwise_ds.append({
            'prompt': prompt,
            'completions': llm_response,
            'labels': labels
        })
    
    stepwise_df = pd.DataFrame(stepwise_ds)
    stepwise_dataset = Dataset.from_pandas(stepwise_df)

    print('final dataset length:', len(stepwise_dataset))
    print(stepwise_dataset[0])
    print(stepwise_dataset[100])
    stepwise_dataset.push_to_hub(STEPWISE_DATASET, private=True)