import re
import Levenshtein

from datasets import *
import pandas as pd

def validate_equations(equations):
    for eq in equations:
        try:
            lhs, rhs = eq.split("=")
            lhs_processed = re.sub(r'(\d+)(\()', r'\1*\2', lhs.strip())
            rhs_processed = re.sub(r'(\d+)(\()', r'\1*\2', rhs.strip())
            
            lhs_val = eval(lhs_processed.strip())
            rhs_val = eval(rhs_processed.strip())
            if abs(lhs_val - rhs_val) > 1e-3:
                return False
        except Exception:
            return False
    return True

def get_equations(answer):
    equations = re.findall(r"<<([^<>]+)>>", answer)
    equations = [equation.replace(' ', '') for equation in equations]
    return equations

def get_distances(ds):
    curr_question = ds[0]['question']
    orig_equations = get_equations(ds[0]['orig_answer'])
    str_orig_equations = '|'.join(orig_equations)

    selected_ds = []

    for d in tqdm(ds):
        question = d['question']
        if question != curr_question:
            curr_question = question
            orig_equations = get_equations(d['orig_answer'])

            if len(orig_equations) == 0:
                orig_equations = [' ']

            str_orig_equations = '|'.join(orig_equations)

        if d['correct'] == False:
            continue

        equations = get_equations(d['llm_answer'])
        if validate_equations(equations) == False:
            continue

        if len(equations) == 0:
            equations = [' ']

        str_equations = '|'.join(equations)
        dist = Levenshtein.distance(str_equations, str_orig_equations)
        dist = dist / len(str_orig_equations)

        d['distance'] = dist
        selected_ds.append(d)

    return selected_ds
        

if __name__ == "__main__":
    ORIG_DATASET = 'ebony59/gsm8k-gen'
    OUTPUT_DATASET = 'ebony59/gsm8k-gen-sel'

    dataset = load_dataset(ORIG_DATASET, split='train')
    ds = dataset.to_pandas().to_dict('records')
    ds = get_distances(ds)
    df = pd.DataFrame(ds)
    new_dataset = Dataset.from_pandas(df)

    print(f"length of checked dataset: {len(new_dataset)}/{len(dataset)}")
    new_dataset.push_to_hub(OUTPUT_DATASET, private=True)

    

