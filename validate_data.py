import re
import Levenshtein

from datasets import *
import pandas as pd

def validate_equations(equations):
    for eq in equations:
        try:
            lhs, rhs = eq.split("=")
            lhs_val = eval(lhs.strip())
            rhs_val = eval(rhs.strip())
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
    avg_distances = []

    for d in tqdm(ds):
        question = d['question']
        if question != curr_question:
            curr_question = question
            orig_equations = get_equations(d['orig_answer'])

            if len(orig_equations) == 0:
                orig_equations = ['None']

        if d['correct'] == False:
            continue

        equations = get_equations(d['llm_answer'])
        if validate_equations(equations) == False:
            continue

        if len(equations) == 0:
            equations = ['None']

        dist = 0
        for i, equation_i in enumerate(equations):
            for j, equation_j in enumerate(orig_equations):
                dist += Levenshtein.distance(equation_i, equation_j)
        dist = dist / (len(equations)*len(orig_equations))
        d['distance'] = dist

    return ds
        

if __name__ == "__main__":
    ORIG_DATASET = 'ebony59/gsm8k-gen'
    OUTPUT_DATASET = 'ebony59/gsm8k-gen-sel'

    dataset = load_dataset(ORIG_DATASET, split='train')
    ds = dataset.to_pandas().to_dict('records')
    ds = get_distances(ds)
    df = pd.DataFrame(ds)
    new_dataset = Dataset.from_pandas(df)
    new_dataset.push_to_hub(OUTPUT_DATASET, private=True)

    

