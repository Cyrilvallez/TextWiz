# TextWiz

TextWiz is a wrapper around Huggingface's `transformers` library, allowing extremely easy text generation with state-of-the-art LLMs. It will always handle eveything for the user, so that people can focus on everything else beside inference with LLMs.

## Install

You can easily install it with pip in your favorite environment:

```sh
pip install textwiz
```

## Usage examples

Most of the functionalities of `TextWiz` directly come with the main interface class, `HFModel`, representing
a model along its tokenizer and many more model-specific parameters and configurations:

```python
import textwiz

model = textwiz.HFModel('mistral-7B')
```

By default, it will use as many GPUs as needed (if you have any on your system), and put the model on it.
Then you can simply perform auto-regresive inference like this:

```python
prompt = 'Chambéry is a nice small city in France. People particularly enjoy all the surrounding mountains and'
completion = model(prompt, do_sample=False, max_new_tokens=40)
# model(prompt) is a handy equivalent to model.generate_text(prompt)

completion
>>> 'the lake.

The city is located in the Rhône-Alpes region, in the Savoie department. It is the capital of the Savoie department and the Haute'
```

As you can see, you only need to provide your prompt! We also highlighted two other arguments, `do_sample` and `max_new_tokens`. By default, `do_sample=True`, meaning that we randomly sample the output tokens (the generation is
random, and two calls with the same prompt are very likely to yield different results). `max_new_tokens` controls the length of the output.

You can also create a conversation with a model. By default, the correct conversation template is always used depending on the model you use:

```python
model = textwiz.HFModel('zephyr-7B-beta')
prompt = 'Hello, who are you?'
conv = model.generate_conversation(prompt, conv_history=None)

print(conv)
>>> """>> User: Hello, who are you?
>> Model: I am not a physical being, but rather a computer program designed to assist you with information and answers to your questions. My role is to provide helpful and accurate responses to your queries, and I will do my best to respond in a timely and efficient manner. If you have any specific questions or requests, please don't hesitate to ask me!"""

conv = model.generate_conversation('What is the purpose of life?', conv_history=conv)

print(conv)
>>> """>> User: Hello, who are you?
>> Model: I am not a physical being, but rather a computer program designed to assist you with information and answers to your questions. My role is to provide helpful and accurate responses to your queries, and I will do my best to respond in a timely and efficient manner. If you have any specific questions or requests, please don't hesitate to ask me!

>> User: What is the purpose of life?
>> Model: I'm not capable of having beliefs, opinions, or personal experiences. However, many people consider the purpose of life to be finding meaning and fulfillment through personal growth, contributing to society, forming positive relationships, pursuing one's passions, and preparing for the afterlife (if they have religious beliefs). Ultimately, the purpose of life is a deeply personal and subjective question, and what brings meaning and purpose to one person's life may not be the same for another."""
```

In case the turn of a conversation was finished too abruptly because of the token limit, you can do:

```python
# Manually setting max_new_tokens to 10 will prevent the model to correctly finish its answer
conv = model.generate_conversation('Hello, who are you?', conv_history=None, max_new_tokens=10)

print(conv)
>>> """>> User: Hello, who are you?
>> Model: I'm not a physical being, but rather"""

# Correctly terminate last turn
conv = model.continue_last_conversation_turn(conv_history=conv)

print(conv)
>>> """>> User: Hello, who are you?
>> Model: I'm not a physical being, but rather a virtual one. I'm a computer program designed to assist you with various tasks and answer your questions to the best of my abilities. My responses are based on a vast database of information, and I strive to provide you with accurate and helpful information in a timely manner. If you have any further questions, please don't hesitate to ask!"""
```

As you can see, the model's answer is now correctly finished. Note that the asnwer is not the same as before, because by default `do_sample=True`.

Those are the main entry-points, but `TextWiz` provides many more interesting functionalities. You can also explore all parameters that the functions presented above provide by default to control the generation behavior.