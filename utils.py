#!usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import os
from typing import Any, List

import aiohttp
import openai
import spacy
import tiktoken
from retry import retry


def setup_tokenizer(model: str = 'gpt-3.5-turbo'):
    """
    - Set up OpenAI tokenizer for GPT models
    Args:
        model: str, the name of the GPT model
                    to set up tokenizer for
    """
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokenizer = tiktoken.encoding_for_model(model)

    return tokenizer


def setup_openai(api_key: str, org_id: str = None):
    """
    - Set up auth for OpenAI
    Args:
        api_key: str, your OpenAI API key
        org_id: str, your organization id 
                     if you registered in OpenAI as organization
    """
    openai.api_key = api_key
    if org_id:
        openai.organization = org_id


def setup_spacy(language_model: str = 'en_core_web_lg'):
    """
    - Load spaCy language model
    """
    nlp = spacy.load(language_model)
    return nlp


def chunk_text(text: str, nlp):
    """
    Chunk the text into sentences using spaCy
    """
    if len(text) == 0:
        return []

    doc = nlp(text, disable=['tagger', 'ner', 'lemmatizer'])
    sent_docs = []
    for sent in doc.sents:
        sent = ' '.join(sent.text.split())
        sent_docs.append(nlp(sent))
    return sent_docs


def filter_incomplete(text: str, nlp=None):
    """
    - Parse the given text into sentences 
      and remove the last sentence if it is partial/incomplete
    """
    if not nlp:
        nlp = setup_spacy()

    spacy_sents = chunk_text(text, nlp)
    # check if the sentence is ending with a closing punctuation
    if not spacy_sents:
        return None
    if not spacy_sents[-1][-1] in ['.', '!', '?']: 
        spacy_sents = spacy_sents[:-1]
    return ' '.join([s_sent.text for s_sent in spacy_sents])


def batchfy(inputs: List, batch_size: int):
    """
    - Batchfy the given inputs list into batches
    Args:
        inputs: List[Any], inputs list to batchfy
        batch_size: int, batch size
    """
    if batch_size > len(inputs):
        return [inputs]
    
    batches = []
    for i in range(0, len(inputs), batch_size):
        batches.append(inputs[i: i + batch_size])
    return batches


# Perform retry if OpenAI returns one of these errors.
# For the detail of each error, see: 
# https://platform.openai.com/docs/guides/error-codes/python-library-error-types
retry_exceptions = (openai.error.APIError, openai.error.Timeout, openai.error.RateLimitError,
                    openai.error.ServiceUnavailableError, openai.error.APIConnectionError)
@retry(exceptions=retry_exceptions, tries=3, delay=10)
def run_gpt_chat(chat_prompt: List, model: str = 'gpt-3.5-turbo', **kwargs):
    """
    - Run GPT-3.5 chat models
    """
    if not isinstance(chat_prompt, list):
        raise Exception('Chat completion requires the list of dictionaries as input prompt')

    # run GPT 3.5 chat for given input prompt
    resp = openai.ChatCompletion.create(
        model=model, 
        messages=chat_prompt,
        **kwargs
    )
    # TODO: Implement a retry mechanism when GPT-3 accidentaly outputs empty string
    # extract response and clean it up
    resp = resp['choices'][0]['message']['content']
    resp = resp.strip()
    return resp


@retry(exceptions=retry_exceptions, tries=3, delay=10)
def run_gpt(prompt: str, model: str = 'text-davinci-003', **kwargs):
    """
    - Run GPT-3 instruct models
    """
    # run GPT for given input prompt
    resp = openai.Completion.create(
        engine=model, 
        prompt=prompt,
        **kwargs
    )
    # TODO: Implement a retry mechanism when GPT-3 accidentaly outputs empty string 
    # extract response and clean it up
    resp = resp.choices[0]['text']
    resp = resp.strip('\n') # strip out a leading '\n' in output
    return resp


async def run_gpt_chat_async(chat_prompt: List, 
                             retry_queue: asyncio.queues.Queue, 
                             store_in_queue: Any,
                             api_key: str,
                             request_url: str = 'https://api.openai.com/v1/chat/completions',
                             model: str = 'gpt-3.5-turbo', 
                             **kwargs):
    """
    - Asyncronous version of `run_gpt_chat` function
    - Instead of using the OpenAI library, directly post the request to endpoint
      so that we can utilize parallel session
    - Also, properly save the request in queue if it fails with specific errors
      so that we can retry later
    Args:
        chat_prompt: List[dict], chat model input prompt list 
        retry_queue: `asyncio.Queue`, a queue that stores the object to retry
        store_in_queue: Any, an object that will be stored in queue when need retry
        api_key: str, OpenAI API key
        request_url: str, a valid URL for OpenAI chat endpoint
        model: str, a valid OpenAI chat model
    """
    if not isinstance(chat_prompt, list):
        raise Exception('Chat completion requires the list of dictionaries as input prompt')
    
    json_body = {
        'model': model,
        'messages': chat_prompt
    }
    json_body.update(kwargs)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=request_url, 
            headers={'Authorization': f'Bearer {api_key}'}, 
            json=json_body
        ) as response:
            response, status_code = await response.json(), response.status
            if status_code == 200:
                response = response['choices'][0]['message']['content']
                response = response.strip()
                return response
            
            elif status_code in (409, 500, 503): 
                # retry on these OpenAI server side errors
                retry_queue.put_nowait(store_in_queue)
                return None

            elif status_code == 429 and 'quota' not in response['error'].get('message', ''):
                # retry only on RateLimitError that has different variations of error messages
                # such as: Rate limit exceeded, server error, model is overloaded, etc.
                # (429 also includes QuotaExceededError that we don't want to retry)
                retry_queue.put_nowait(store_in_queue)
                return None

            else:
                # For other errors, raise Exception
                message = f"{response['error'].get('type', '')}: {response['error'].get('message', '')}"
                raise Exception(message)
            

async def run_gpt_async(prompt: str, 
                        retry_queue: asyncio.queues.Queue, 
                        store_in_queue: Any,
                        api_key: str,
                        request_url: str = 'https://api.openai.com/v1/completions',
                        model: str = 'text-davinci-003', 
                        **kwargs):
    """
    - Asyncronous version of `run_gpt` function
    - Basically the same operation as `run_gpt_chat_async` function, 
      just has different way to parse the output accordingly
    Args:
        prompt: str, text model prompt for instruct models
        retry_queue: `asyncio.Queue`, a queue that stores the object to retry
        store_in_queue: Any, an object that will be stored in queue when need retry
        api_key: str, OpenAI API key
        request_url: str, a valid URL for OpenAI chat endpoint
        model: str, a valid OpenAI chat model
    """
    json_body = {
        'model': model,
        'prompt': prompt
    }
    json_body.update(kwargs)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=request_url, 
            headers={'Authorization': f'Bearer {api_key}'}, 
            json=json_body
        ) as response:
            response, status_code = await response.json(), response.status
            if status_code == 200:
                response = response.choices[0]['text']
                response = response.strip()
                return response
            
            elif status_code in (409, 500, 503): 
                # retry on these OpenAI server side errors
                retry_queue.put_nowait(store_in_queue)
                return None

            elif status_code == 429 and 'quota' not in response['error'].get('message', ''):
                # retry only on RateLimitError that has different variations of error messages
                # such as: Rate limit exceeded, server error, model is overloaded, etc.
                # (429 also includes QuotaExceededError that we don't want to retry)
                retry_queue.put_nowait(store_in_queue)
                return None

            else:
                # For other errors, raise Exception
                message = f"{response['error'].get('type', '')}: {response['error'].get('message', '')}"
                raise Exception(message)