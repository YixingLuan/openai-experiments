from utils import openai_error_handler
from prompts.summarization import summarize
from smart_open import open as sopen
import json
from typing import List
import argparse


def parse_json(input_file: str):
    # Expects jsonl file (each line is a json dict with {'text': '...'})
    with sopen(input_file, 'r') as f:
        lines = f.readlines()
    texts = [json.loads(l.strip()) for l in lines]
    return texts


def write_json(output_file: str, input_texts: List[str], outout_texts: List[str]):
    with sopen(output_file, 'w') as f:
        for in_text, out_text in zip(input_texts, output_texts):
            d = {
                'text': in_text,
                'output': out_text
            }
            f.write(json.dumps(d))
            f.write('\n')
        

def main(**kwargs)
    openai_key = kwargs.pop('api_key')
    setup_openai(openai_key)

    task = kwargs.pop('task')
    model = kwargs.pop('model')

    input_file = kwargs.pop('input_file')
    output_file = kwargs.pop('output_file')

    if task == 'summarization':
        texts = parse_json(input_file)
        if len(texts) > 1:
            # TODO: add batch processing in async
            pass
        else:
            resp = summarize(texts[0], model, **kwargs)
            write_json(output_file, texts, [resp])

    else:
        raise NotImplementedError(f'The specified task {task} is currently not supported.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api_key',
                        dest='api_key',
                        required=True,
                        help='OpenAI API key')
    parser.add_argument('--task',
                        dest='task',
                        required=True,
                        choices=['summarization'], # TODO: expand this list accordingly
                        help='A task to perform')
    parser.add_argument('--model',
                        dest='model',
                        default='gpt-3.5-turbo'
                        help='OpenAI GPT model to use')
    parser.add_argument('--input_file',
                        dest='input_file',
                        required=True,
                        help='Input jsonl file that contains input context')
    parser.add_argument('--output_file',
                        dest='output_file',
                        required=True,
                        help='Output jsonl file path to write results')
    parser.add_argument('--temperature',
                        dest='temperature',
                        default=0.7,
                        type=float,
                        help='GPT model temperature')
    parser.add_argument('--max_tokens',
                        dest='max_tokens',
                        default=100,
                        type=int,
                        help='Max number of tokens to generate')
    args = parser.parse_args()

    main(**args)

