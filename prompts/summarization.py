#!usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import asyncio

from utils import (filter_incomplete, batchfy, run_gpt, 
                   run_gpt_chat, setup_tokenizer, 
                   run_gpt_async, run_gpt_chat_async)


def limit_prompt(prompt, accepted_length, tokenizer):
    """
    - Limit the input prompt length based on token count
      so that we don't go over model limit
    - This will only work when prompting command is in the head
      (prompt format is: request line followed by context)
    """
    return tokenizer.decode(tokenizer.encode(prompt)[:accepted_length])


def construct_chat_prompt(text: str, max_tokens: int, tokenizer):
    """
    - Set up chat prompt for summarization
    """
    # TODO: Adjust max length based on models 
    accepted_length = 4097 - max_tokens - 1 # -1 is for '\n' in the end (for instruct models)

    system_prompt = 'You are a precise and descriptive text summarizer.'
    prompt = f'Please summarize the following text in English:\n{text}'

    accepted_length -= len(tokenizer.encode(system_prompt))
    # Limit input length so that we don't go over model limit
    prompt = limit_prompt(prompt, accepted_length, tokenizer)

    prompts = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]

    return prompts


def construct_prompt(text: str, max_tokens: int, tokenizer):
    """
    - Set up normal prompt for summarization
    """
    prompt = f'Please summarize the following text in English:\n{text}'

    # TODO: Adjust max length based on models 
    accepted_length = 4097 - max_tokens - 1 # -1 is for '\n' in the end (for instruct models)
    accepted_length -= len(tokenizer.encode('\n\nSummary in English: '))
    # Limit input length so that we don't go over model limit
    prompt = limit_prompt(prompt, accepted_length, tokenizer)

    prompt += '\n\nSummary: '

    return prompt


def summarize(text: str, model: str = 'gpt-3.5-turbo', **kwargs):
    """
    - Summarize the given text
    Args:
        text: str, input text to summarize
        model: str, GPT model name
    """
    max_tokens = kwargs.get('max_tokens', 300)
    tokenizer = setup_tokenizer(model)

    if model in ['gpt-3.5-turbo', 'gpt-4']: # chat completion
        prompts = construct_chat_prompt(text, max_tokens, tokenizer)
        response = run_gpt_chat(prompts, model, **kwargs)
 
    else: # text completion
        prompt = construct_prompt(text, max_tokens, tokenizer)
        response = run_gpt(prompt, model, **kwargs)

    response = filter_incomplete(response)

    return response


async def summarize_async(text_batch: List[str], api_key: str, model: str, tokenizer, **kwargs):
    """
    - The function that performs actual async calls to GPT 
      to do summarization
    Args:
        text_batch: List[str], a batch of input text to summarize
        api_key: str, OpenAI API key
        model: str, model name
        tokenizer: GPT tokenizer
    """
    retry_queue = asyncio.Queue()
    text_batch = iter(text_batch)
    summaries = {}

    max_tokens = kwargs.get('max_tokens', 300)

    while True:
        if not retry_queue.empty():
            # sleep 10 seconds before retry
            await asyncio.sleep(10)
            input_text = retry_queue.get_nowait()
        else:
            try:
                input_text = next(text_batch)
            except StopIteration:
                break

        if model in ['gpt-3.5-turbo', 'gpt-4']: # chat completion
            prompts = construct_chat_prompt(input_text, max_tokens, tokenizer)
            task = asyncio.create_task(
                run_gpt_chat_async(prompts, retry_queue, input_text, api_key, **kwargs)
            )

        else: # text completion
            prompt = construct_prompt(input_text, max_tokens, tokenizer)
            task = asyncio.create_task(
                run_gpt_async(prompt, retry_queue, input_text, api_key, **kwargs)
            )
        summaries[input_text] = task

        await asyncio.sleep(0.001)

    return summaries


async def batch_summarize(text_batch: List[str], api_key: str, model: str = 'gpt-3.5-turbo', **kwargs):
    """
    - Summarize the input batch of text in async process
    Args:
        text_batch: List[str], list of input texts to summarize
        api_key: str, OpenAI API key
        model: str, GPT model name
    """
    # 1. set up tokenizer
    tokenizer = setup_tokenizer(model)

    # 2. batchfy the input test list
    #    (This is useful to avoide exceeding the rate limit of OpenAI API. 
    #     For the detail of rate limit, 
    #     see: https://platform.openai.com/docs/guides/rate-limits/overview)
    input_batches = batchfy(text_batch, batch_size=1000)

    # 3. summarize 
    results = []
    sleep = True if len(input_batches) > 1 else False
    for batch in input_batches:
        summaries = await summarize_async(text_batch, api_key, model, tokenizer, **kwargs)
        summaries = [await summaries[input_text] for input_text in batch]
        summaries = [filter_incomplete(summary) for summary in summaries]
        results.extend(summaries)

        if sleep: # sleep 10 secs to make sure we don't hit rate limit
            await asyncio.sleep(10)

    return results