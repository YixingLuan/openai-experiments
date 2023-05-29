#!usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import os
from typing import Any, List

import aiohttp
import openai
from google.cloud import secretmanager
from retry import retry


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


def openai_error_handler(error):
    """
    - Depending on the OpenAI error type, 
      return different error messages and status codes
    Args:
        error: `openai.error`, error from OpenAI python package
    """
    parse_openai_error = lambda x: (x.http_status, x.json_body['error']['message'])

    if type(error) == openai.error.OpenAIError:
        status_code, message = parse_openai_error(error)
        return f'openai.error.OpenAIError: {message}', status_code
    
    elif type(error) == openai.error.APIError:
        status_code, message = parse_openai_error(error)
        return f'openai.error.APIError: {message}', status_code
    
    elif type(error) == openai.error.TryAgain:
        status_code, message = parse_openai_error(error)
        return f'openai.error.TryAgain: {message}', status_code
    
    elif type(error) == openai.error.Timeout:
        status_code, message = parse_openai_error(error)
        return f'openai.error.Timeout: {message}', status_code
    
    elif type(error) == openai.error.APIConnectionError:
        status_code, message = parse_openai_error(error)
        return f'openai.error.APIConnectionError: {message}', status_code
    
    elif type(error) == openai.error.InvalidRequestError:
        status_code, message = parse_openai_error(error)
        return f'openai.error.InvalidRequestError: {message}', status_code
    
    elif type(error) == openai.error.AuthenticationError:
        status_code, message = parse_openai_error(error)
        return f'openai.error.AuthenticationError: {message}', status_code
    
    elif type(error) == openai.error.PermissionError:
        status_code, message = parse_openai_error(error)
        return f'openai.error.PermissionError: {message}', status_code
    
    elif type(error) == openai.error.RateLimitError:
        status_code, message = parse_openai_error(error)
        return f'openai.error.RateLimitError: {message}', status_code
    
    elif type(error) == openai.error.ServiceUnavailableError:
        status_code, message = parse_openai_error(error)
        return f'openai.error.ServiceUnavailableError: {message}', status_code
    
    elif type(error) == openai.error.InvalidAPIType:
        status_code, message = parse_openai_error(error)
        return f'openai.error.InvalidAPIType: {message}', status_code
    
    elif type(error) == openai.error.SignatureVerificationError:
        status_code, message = parse_openai_error(error)
        return f'openai.error.SignatureVerificationError: {message}', status_code
    
    else:
        return f'{error}', 500


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