#!usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
from utils import setup_tokenizer, filter_incomplete run_gpt_chat, run_gpt


def summarize(text: str, model: str = 'gpt-3.5-turbo', **kwargs):
    """
    - Summarize the given text
    Args:
        text: str, input text to summarize
        model: str, GPT model name
    """
    def limit_prompt(prompt, accepted_length, tokenizer):
        # This will only work when prompting command is in the head
        # (prompt format is: request line followed by context)
        return tokenizer.decode(tokenizer.encode(prompt)[:accepted_length])
    max_tokens = kwargs.get('max_tokens', 300)
    # TODO: Adjust max length based on models 
    accepted_length = 4097 - max_tokens - 1 # -1 is for '\n' in the end (for instruct models)
    tokenizer = setup_tokenizer(model)

    prompt = f'Please summarize the following text in English:\n{text}'

    if model in ['gpt-3.5-turbo', 'gpt-4']: # chat completion
        system_prompt = 'You are a precise and descriptive text summarizer.'
        accepted_length -= len(tokenizer.encode(system_prompt))
        # Limit input length so that we don't go over model limit
        prompt = limit_prompt(prompt, accepted_length, tokenizer)
        prompts = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]
        response = run_gpt_chat(prompts, model, **kwargs)
 
    else: # text completion
        accepted_length -= len(tokenizer.encode('\n\nSummary in English: '))
        # Limit input length so that we don't go over model limit
        prompt = limit_prompt(prompt, accepted_length, tokenizer)
        prompt += '\n\nSummary: '
        response = run_gpt(prompt, model, **kwargs)

    response = filter_incomplete(response)

    return response


def batch_summarize(text_batch: List[str], model: str = 'gpt-3.5-turbo', **kwargs):
    """
    - Summarize the input batch of text in async process
    Args:
        text: str, input text to summarize
        model: str, GPT model name
    """
    # TODO
    pass