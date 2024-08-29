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

# State of the field

The most commonly used Python package for using LLMs is `transformers` [@transformers]. However, as explained, every model usually has its own implementation/usage details, making it hard to run the same text generation pipeline out-of-the-box for different models. Common pitfalls are also the responsability of the users to navigate. `TextWiz` goes one big step beyond, in order to provide one common interface and alleviate all common issues.

# References