# TextWiz

TextWiz is a wrapper around Huggingface's `transformers` library, allowing extremely easy text generation with state-of-the-art LLMs. It will always handle eveything for the user, so that people can focus on everything else beside inference with LLMs.  
All the boilerplate (`pytorch` devices, prompt/conversation formatting, batching, memory overflows, etc...) that would otherwise arise all the time when performing inference with LLMs is automated by `TextWiz`. Users only need to provide their prompts as `strings`, and feed that into the appropriate `HFModel` class along with generation parameters.

## Install

You can easily install it with pip in your favorite environment:

```sh
pip install textwiz
```

## Hardware requirements

Modern deep learning models are so big that it has become vital to run them on specific hardware (usually GPUs). As `TextWiz` focuses on easily serving such models, it is no exception to the rule. Even if most of the library **can** be run on CPU only, doing so is going to be **excruciatingly slow** in almost all cases, and some functionalities may be more fragile.  

However, as long as you have at least 1 GPU available (visible with `nvidia-smi`), `TextWiz` will always use it by default and handle all communication and memory management with it automatically. For bigger models, more GPU(s) will be required.

## Usage examples

### Causal language modeling

Most of the functionalities of `TextWiz` directly come with the main interface class, `HFCausalModel`, representing
a model along its tokenizer and many more model-specific parameters and configurations:

```python
import textwiz

model = textwiz.HFCausalModel('mistral-7B')
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

As you can see, you only need to provide your prompt as a string! We also highlighted two other arguments, `do_sample` and `max_new_tokens`. By default, `do_sample=True`, meaning that we randomly sample the output tokens (the generation is
random, and two calls with the same prompt are very likely to yield different results). `max_new_tokens` controls the length of the output.

You can also create a conversation with a model. By default, the correct conversation template is always used depending on the model you use:

```python
model = textwiz.HFCausalModel('zephyr-7B-beta')
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

If the turn of a conversation was finished too abruptly because of the `max_new_tokens` limit, you can do:

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

As you can see, the model's answer is now correctly finished. Note that the answer is not the same as before, because by default `do_sample=True`.

Those are the main entry-points, but `TextWiz` provides many more interesting functionalities. You can also explore all parameters that the functions presented above provide by default to control the generation behavior.


### Text embeddings

`TextWiz` also provides a convenient way to create text embeddings using LLMs. Given some texts, simply create the embeddings in the following way:

```python
import textwiz

model = textwiz.HFEmbeddingModel('SFR-Embedding-Mistral')
chunks_to_embed = ["Dummy text", "some other dummy text", "more and more text"]
embeddings = model(chunks_to_embed) # each row is the vector representing given text
```
