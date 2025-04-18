import torch
import datasets
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from tqdm import tqdm
from generate_data import generate_solution

def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_dataset: datasets.Dataset,
    samples='all',
    verbose=False,
) -> float:

    questions = eval_dataset['question']
    answers = eval_dataset['answer']

    correct = 0

    if samples != 'all':
        questions = questions[:samples]
        answers = answers[:samples]

    for question, answer in tqdm(zip(questions, answers), total=len(questions)):
        gt_answer = answer.split("####")[1].lstrip(' ').strip()
        if verbose:
            print('question:', question)
            print('gt_answer:', answer)
        responses = generate_solution(model, tokenizer, question, 1, verbose)
        response = responses[0]
        response_answer = response.split("####")[1].lstrip(' ').strip()
        if gt_answer == response_answer:
            correct += 1

    return correct / len(questions)

if __name__ == "__main__":
    MODEL = "ebony59/phi3.5-gsm8k-SFT"
    DATASET = "openai/gsm8k"

    samples = 'all'
    verbose = False
    
    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    original_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    original_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    
    ds = datasets.load_dataset(DATASET, 'main')

    print('evaluate our score')
    our_score = evaluate_model(model, tokenizer, ds['test'], samples, verbose)
    print('our_score:', our_score)

    print('evalute their score')
    their_score = evaluate_model(original_model, original_tokenizer, ds['test'], samples, verbose)
    print('their_score:', their_score)