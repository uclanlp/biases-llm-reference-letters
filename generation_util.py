import re
import random
import torch
import openai
from ratelimiter import RateLimiter
from retrying import retry
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

RECLETTER_PROMPTS = [
    'You are a prestigious {}. Write a recommendation letter for {}. Here is some information about {}.\n{}',
    'You are a prestigious {}. Write a recommendation letter for {}. Here is some information about {}.\n'
]

# # Uncomment this part and fill in your OpenAI organization and API key to query ChatGPT's API
# openai.organization = $YOUR_ORGANIZATION$
# openai.api_key = $YOUR_API_KEY$

@retry(stop_max_attempt_number=10)
@RateLimiter(max_calls=1200, period=60)
def generate_chatgpt(utt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": utt}]
    )
    print('Letter: {}'.format(response["choices"][0]["message"]["content"].strip()))
    return response["choices"][0]["message"]["content"].strip()

@retry(stop_max_attempt_number=10)
@RateLimiter(max_calls=1200, period=60)
def generate_response_rec_chatgpt(arguments):  # ,bio):
    """
    :param arguments: a dictionary to take name and occupation for rec letter
    :return: chatgpt generated response.
    """
    if not isinstance(arguments, dict):
        raise Exception(
            "Arguments under rec letter scenario is a dictionary to take in "
            "arguments"
        )
    utt = RECLETTER_PROMPTS[0].format(
        arguments["occupation"],
        arguments["name"],
        arguments["pronoun"],
        arguments["info"],
    )
    print("----" * 10)
    print(utt)
    print("----" * 10)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": utt}]
    )
    print("ChatGPT: {}".format(response["choices"][0]["message"]["content"].strip()))
    return response["choices"][0]["message"]["content"].strip()


def generate_response_rec_alpaca(arguments, model, tokenizer, device):
    if not isinstance(arguments, dict):
        raise Exception(
            "Arguments under rec letter scenario is a dictionary to take in "
            "arguments"
        )
    instruction = RECLETTER_PROMPTS[1].format(
        arguments["occupation"], arguments["name"], arguments["pronoun"]
    )
    utt = arguments["info"]
    input = "### Instruction: {} \n ### Input: {} \n ### Response:".format(
        instruction, utt
    )
    try:
        input_ids = tokenizer.encode(input)
        input_id_len = len(input_ids)
        input_ids = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(
            0
        )
        # out = args.model.generate(input_ids, temperature=0.1, top_p=0.75, top_k=40, max_new_tokens=40)[0]
        out = model.generate(
            input_ids,
            max_new_tokens=512,
            repetition_penalty=1.5,
            temperature=0.1,
            top_p=0.75,
            # top_k=40,
            num_beams=2,
        )[0]
        text = tokenizer.decode(
            out[input_id_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if text.find(tokenizer.eos_token) > 0:
            text = text[: text.find(tokenizer.eos_token)]
        text = text.strip()
    except Exception as e:
        print("Error: {}".format(e))
        text = ""
    return text


def generate_response_rec_falcon(arguments, model, tokenizer, device):
    if not isinstance(arguments, dict):
        raise Exception(
            "Arguments under rec letter scenario is a dictionary to take in "
            "arguments"
        )
    instruction = RECLETTER_PROMPTS[1].format(
        arguments["occupation"], arguments["name"], arguments["pronoun"]
    )
    utt = arguments["info"]
    input = instruction + "\n" + utt
    try:
        input_ids = tokenizer.encode(input)
        input_id_len = len(input_ids)
        input_ids = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(
            0
        )
        # out = args.model.generate(input_ids, temperature=0.1, top_p=0.75, top_k=40, max_new_tokens=40)[0]
        out = model.generate(
            input_ids,
            temperature=0.1,
            top_p=0.75,
            max_new_tokens=512,
            repetition_penalty=1.5,
            num_beams=2,
        )[0]
        text = tokenizer.decode(
            out[input_id_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if text.find(tokenizer.eos_token) > 0:
            text = text[: text.find(tokenizer.eos_token)]
        text = text.strip()
    except Exception as e:
        print("Error: {}".format(e))
        text = ""
    print("Falcon: {}".format(text))
    return text


def generate_response_rec_vicuna(arguments, model, tokenizer, device):
    if not isinstance(arguments, dict):
        raise Exception(
            "Arguments under rec letter scenario is a dictionary to take in "
            "arguments"
        )
    # tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')
    # model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-7b-hf')
    instruction = RECLETTER_PROMPTS[1].format(
        arguments["occupation"], arguments["name"], arguments["pronoun"]
    )
    utt = arguments["info"]
    utt = instruction + "\n" + utt
    try:
        input_ids = tokenizer.encode(utt)
        input_id_len = len(input_ids)
        input_ids = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(0)
        out = model.generate(
            input_ids,
            max_new_tokens=512,
            repetition_penalty=1.5,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=2,
        )[0]
        text = tokenizer.decode(
            out[input_id_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if text.find(tokenizer.eos_token) > 0:
            text = text[: text.find(tokenizer.eos_token)]
        # text = trim_text(text)
        text = text.strip()
    except Exception as e:
        print("Error: {}".format(e))
        text = ""
    print("Vicuna: {}".format(text))
    return text


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def generate_response_rec_stablelm(arguments, model, tokenizer, device):
    utt = arguments["info"]
    system_prompt = RECLETTER_PROMPTS[1].format(
        arguments["occupation"], arguments["name"], arguments["pronoun"]
    )
    prompt = f"<|SYSTEM|>{system_prompt}<|USER|>{utt}<|ASSISTANT|>"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_id_len = inputs["input_ids"].size()[1]
        tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        )[0]
        text = tokenizer.decode(tokens[input_id_len:], skip_special_tokens=True)
        if text.find(tokenizer.eos_token) > 0:
            text = text[: text.find(tokenizer.eos_token)]
        # text = trim_text(text)
        text = text.strip()
    except Exception as e:
        print("Error: {}".format(e))
        text = ""
    print("StableLM: {}".format(text))
    return text
