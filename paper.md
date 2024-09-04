---
title: 'TextWiz: An even simpler way to perform inference with LLMs.'
tags:
  - Python
  - Deep learning
  - Text generation
  - Text embedding
  - LLMs
authors:
  - name: Cyril Vallez
    orcid: 0009-0005-7991-233X
    affiliation: 1
affiliations:
 - name: HES-SO Valais-Wallis, Switzerland
   index: 1
date: 29 August 2024
bibliography: paper.bib
---

# Summary

Large Language Models (LLMs) represent a breakthrough in artificial intelligence, enabling machines to comprehend, generate, and manipulate human language with unprecedented accuracy and sophistication. Trained on extensive datasets, these models can perform an array of language-related tasks, from nuanced conversation and content generation to translation, summarization, and beyond. For this reason, researchers with various background started to use these models in an always broader research areas. However, for people with limited knowledge of the deep learning field itself (and for the more experienced researchers as well), many problems and pitfalls (memory, formatting, stopping,...) will arise when trying to use these models outside some very simple tutorial found online.

# Statement of need

`TextWiz` is a wrapper around Huggingface's `transformers` library [@transformers] and `pytorch` [@pytorch], allowing extremely easy text generation with state-of-the-art LLMs. It will abstract eveything for the user, so that people can focus on everything else beside fighting to make their model work correctly. It provides a very simple and intuitive class-based API wrapping a model along with its corresponding tokenizer, so that any model supported can be used in the exact same way, without taking care of the specific internal details of each model.  

`TextWiz` was designed during the benchmarking of code generation capabilities of a large range of open-source models. As such, it is designed to be used by both novice and seasoned researchers wishing to easily run (expensive) benchmarks on a large array of models at the same time, without worrying about the specifics of each model (and common problems to all models). It also provides convenient functions allowing to develop web demos using `gradio` [@gradio] very simply, to demonstrate models and play with them in a browser. Its design will enable novices and experts alike to focus on everything else except the boilerplate of inference.  

A major issue on most inference pipelines (benchmarking different models on the same dataset) is memory management. Indeed, it is very easy to overflow the memory and crash the program with current extremely large models. If one is not extremely careful about memory usage and has no idea of the memory footprint required for inference itself (not to be confused with the memory needed to load the model), a long struggle will ensue in which the user is likely to have a painful and time-consuming battle to try to somehow maximize memory usage without overflowing it. `TextWiz` solves this problem with automatic and extremely accurate memory estimation of inference for supported models and configurations (i.e. dtype, attention algorithms,...). A simple memory estimation (in GiB) for a given model, input size and number of additional new tokens can also be obtained through the CLI `textwiz-memory` in the following way:

```sh
textwiz-memory llama3-8B 1122 512
>>> 0.95424698205538
```

# State of the field

The most commonly used Python package for using LLMs is `transformers` [@transformers]. However, as explained, every model usually has its own implementation/usage details, making it hard to run the same text generation pipeline out-of-the-box for different models. Common pitfalls are also the responsability of the users to navigate. `TextWiz` goes one big step beyond, in order to provide one common interface and alleviate all common issues. It is also much simpler to use for beginners. Here is a very simple code snippet required to have a conversation with  using `transformers`:

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

Now, to do the *exact same* thing using `TextWiz`:

```py
import textwiz

model = textwiz.HFCausalModel('llama3-8B-instruct')
system_prompt = "You are a pirate chatbot who always responds in pirate speak!"
conv = model.generate_conversation('Who are you?', conv_history=None, system_prompt=system_prompt, max_new_tokens=256, temperature=0.6, top_p=0.9)
print(conv.model_history_text[-1])
```

As one can observe, it is mush easier with `TextWiz`, and beginners do not have to worry about any details, be it proper tokenization, chat template, best dtype, torch device, end of sequence (eos) tokens, etc...

# References