import torch
import datasets
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from more_itertools import chunked

from tqdm import tqdm

def generate_message(question):
    system_prompt = """
    You are an AI assistant to solve maths problems. For each question, write a step-by-step solution and give the final answer in format: solution\n#### final_answer
    Only include the solution and answer, do not include any other descriptions.
    
    Example:
    *** Instruction:
    Question:
    Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? 
    
    *** Your answer should be:
    Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
    #### 72
    """

    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"},
      {"role": "assistant", "content": 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72'},
      {"role": "user", "content": question}
    ]

    return messages
    

def generate_solution(
    pipe,
    questions,
    k=10,
    verbose=False,
) -> datasets.Dataset:
    
    prompts = [generate_message(q) for q in questions for i in range(k)]
    temp_ds = datasets.Dataset.from_dict({"messages": prompts})

    generation_args = {
      "max_new_tokens": 500,
      "return_full_text": False,
      "do_sample": True,
      "temperature": 0.7,
    }

    with torch.inference_mode():
        outputs = pipe(temp_ds['messages'], **generation_args)
    responses = []

    for i, output in enumerate(outputs):
        text = output[0]['generated_text']
        responses.append(text)

    return responses
        

def generate_synthetic_data(model, tokenizer, ds, k=10, samples='all', verbose=False):
    dataset = []

    questions = ds['question']
    answers = ds['answer']

    chunk_size = 256
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
        batch_size=128
    )

    for chunk_idx, (question_chunk, answer_chunk) in enumerate(tqdm(zip(question_chunks, answer_chunks), total=len(question_chunks))):
        print(f"Processing chunk {chunk_idx}")
        gt_answers = [answer.split("####")[1].lstrip(' ').strip() for answer in answer_chunk]
        
        responses = generate_solution(pipe, question_chunk, k, verbose)

        for i, response in enumerate(responses):
            question = question_chunk[i // k]
            answer = answer_chunk[i // k]
            gt_answer = gt_answers[i // k]
            try:
                response_answer = response.split("####")[1].lstrip(' ').strip()
                if gt_answer == response_answer:
                    correct = True
                else:
                    correct = False
                dataset.append({'question': question, 'gt_answer': gt_answer, 'orig_answer': answer,'llm_answer': response, 'correct': correct})
            except:
                continue

        temp_df = pd.DataFrame(dataset)
        temp_df.to_csv(f'/workspace/syn_data.csv')
    return dataset

if __name__ == "__main__":
    MODEL = "ebony59/phi3.5-gsm8k-SFT"
    # MODEL = "microsoft/Phi-3.5-mini-instruct"
    DATASET = "openai/gsm8k"
    
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
    
    syn_data = generate_synthetic_data(
        model, 
        tokenizer, 
        ds['train'],
        k=10,
        samples='all',
        verbose=False
    )

    df = pd.DataFrame(syn_data)
    df.to_csv(f'/workspace/syn_data.csv')

    dataset = datasets.Dataset.from_pandas(df)

    dataset.push_to_hub('ebony59/gsm8k-gen', private=True)