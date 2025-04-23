import torch
import datasets
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from more_itertools import chunked

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

    chunk_size = 128
    question_chunks = list(chunked(questions, chunk_size))
    answer_chunks = list(chunked(answers, chunk_size))

    print(f"Number of chunks: {len(question_chunks)}, chunk_size: {chunk_size}")

    if samples != 'all':
        questions = questions[:samples]
        answers = answers[:samples]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=64
    )

    correct = 0

    for chunk_idx, (question_chunk, answer_chunk) in enumerate(tqdm(zip(question_chunks, answer_chunks), total=len(question_chunks))):
        print(f"Processing chunk {chunk_idx}")
        gt_answers = [answer.split("####")[1].lstrip(' ').strip() for answer in answer_chunk]
        
        responses = generate_solution(pipe, question_chunk, 1, verbose)

        for i, response in enumerate(responses):
            question = question_chunk[i]
            gt_answer = gt_answers[i]
            try:
                response_answer = response.split("####")[1].lstrip(' ').strip()
                if gt_answer == response_answer:
                    correct += 1
            except:
                continue

    return correct / len(questions)

if __name__ == "__main__":
    MODEL = "ebony59/phi3.5-gsm8k-syn-FT-2"
    DATASET = "openai/gsm8k"

    samples = 'all'
    verbose = False
    
    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    ds = datasets.load_dataset(DATASET, 'main')

    score = evaluate_model(model, tokenizer, ds['test'], samples, verbose)
    print('score:', score)