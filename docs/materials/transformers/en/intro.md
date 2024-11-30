<div class="content-2columns" markdown>
![](../assets/imgs/math-mandelbrot.png){: .rounded-title-img}

# A Step-by-Step Guide to Transformers: Understanding How Neural Networks Process Texts and How to Program Them
</div>

## Introduction

This guide provides a pathway to understanding how the most widely used neural network in natural language processing (*transformer*) actually works. It follows the theoretical explanations of selected chapters from a well-regarded book on the subject. It also proposes learning Python programming along with the basics of a library called PyTorch, which enables neural networks to be programmed, trained, and run on GPUs. As a culmination, an existing implementation of the transformer programmed with PyTorch is studied. The ultimate goal is to modify this code to experiment with a simple problem involving human language. The idea is to gain solid knowledge for tackling more complex tasks later, rather than creating something flashy to showcase immediately.

!!! note "Note"

    This page outlines a self-learning roadmap for understanding transformers. It links to documents hosted on other pages of this website. Thus, the collection can be considered a complete guide to assist you on your journey. However, you may have come to these pages from another source (e.g., a specific course) that suggests a different way of using the various contents. In that case, follow the recommendations and plan provided by that source instead of those proposed here.

Some content can be studied in parallel. While learning about neural models, you can start exploring [Python](pytorch.md#python), NumPy, and later [PyTorch](pytorch.md#pytorch) after the first two. You can also review the [elements of algebra, calculus, and probability](#prerequisite) that you might have forgotten. Studying the transformer's code should only be undertaken after thoroughly assimilating all these prior concepts.

## Study Manual

To understand neural networks mathematically and conceptually, we will rely on the third edition (still unfinished) of the book "[Speech and Language Processing][libroarch]" by Dan Jurafsky and James H. Martin. Sections of this guide indicate which chapters and sections are relevant for our purposes. **Important**: Since the online version of the book is unfinished and periodically updated, not only with new content but also with restructurings and section relocations, this guide includes links and references to an [archived version of the book][libroarch] that may not correspond to the latest version (available [here][libro]).

[libro]: https://web.stanford.edu/~jurafsky/slp3/
[libroarch]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/

## Why a Deep Dive Approach?

At first glance, writing a program that uses machine learning models seems straightforward. For example, the following lines of code use a language model based on a transformer to complete a given text:

```python
from transformers import pipeline
generator = pipeline('text-generation', model = 'gpt2')
generator("Hello, I'm a language model and", max_length = 30, num_return_sequences=3)
```

While high-level libraries are immensely important in certain contexts, if you only use the code above:

- You won't understand how the model actually works.
- You won't be able to create other models to experiment with different problems.
- You won't know how to train your own model or what factors influence training quality or duration.
- You won't understand other neural models used in natural language processing.
- Ultimately, you’ll view your program as a black box performing magical tasks.

This guide aims to help you open that black box and understand its workings thoroughly.

## Content Sequencing

The following table shows a sequence of the guide's content along with indicative time estimates for each part.

| Step | Content | Estimated Time :octicons-stopwatch-24: | Notes |
| -------- | ------- | --------------- | ------------- |
| 1 | [Introduction](intro.md) | 10 minutes | This page! |
| 2 | [Mathematical Concepts](intro.md#prerequisite) | 5 hours | Refer to the links in this section only if you need to refresh your knowledge of mathematical concepts. |
| 3 | [Regressors](regresor.md) | 4 hours | This document introduces a machine learning model that is not usually categorized as a neural network but helps introduce most of the key ideas relevant for discussing neural networks. |
| 4 | [Learning PyTorch](pytorch.md) | 5 hours | Going beyond equations and learning to implement the different models we will study is fundamental to fully understanding them. This page links to resources for learning (or reviewing) Python and PyTorch. Invest time here before advancing to theoretical content to better understand the implementations. |
| 5 | [Regressor Implementation](implementacion.md#code-regressor) | 4 hours | Examine PyTorch code for implementing logistic and softmax regressors. Use debugging tools as explained [here][debug] to step through the code: analyze variable values and types, tensor shapes, and ensure you understand what each dimension represents. |
| 6 | [Non-contextual Embeddings](embeddings.md) | 4 hours | Obtaining non-contextual embeddings is an immediate application of logistic regressors, showcasing the potential of self-supervised learning. |
| 7 | [Skip-gram Implementation](implementacion.md#code-skipgrams) | 2 hours | Analyze PyTorch code for skip-gram implementation. Use debugging tools as explained [here][debug] to step through the code. By this point, it is advisable to start familiarizing yourself with PyTorch before proceeding further. |
| 8 | [Feedforward Networks](ffw.md) | 3 hours | This section introduces the neural network concept and creates a very basic language model with feedforward networks. |
| 9 | [Feedforward Network Implementation](implementacion.md#code-ffw) | 1 hour | Explore the code for implementing a simple feedforward-based language model. |
| 10 | [Transformers and Attention Models](attention.md) | 6 hours | All previous concepts prepare you to delve into the transformer architecture. This page focuses on the transformer’s *decoder* part, used in language models. |
| 11 | [Transformer Code Implementation](implementacion.md#code-transformer) | 6 hours | Analyze the general transformer implementation and a decoder-based language model. This code is more complex than those you studied before. |
| 12 | [Additional Aspects of Transformers](attention2.md) | 4 hours | This page introduces the transformer’s *encoder* part and its potential uses, both standalone and paired with a decoder. |
| 13 | [Named Entity Recognizer Implementation](implementacion.md#code-transformer) | 1 hour | Analyze the code for implementing a named entity recognizer based on an encoder. |
| 14 | [GPT-2 Model Implementation](implementacion.md#code-mingpt) | 4 hours | Optionally analyze the code for implementing a language model capable of loading and using GPT-2. |
| 15 | [Speech](speech.md) | 4 hours | This content is optional as it shifts to the domain of speech processing. |

[debug]: pytorch.md#debug

In each section, the :octicons-book-24: icon highlights essential links, whether to book chapters for reading or code for exploration.

## Prerequisite Mathematical Concepts {#prerequisite}

Basic algebra, calculus, and probability concepts needed for natural language processing can be found in the "Linear Algebra," "Calculus" (including "Automatic differentiation"), and "Probability and Statistics" sections of [:octicons-book-24:][cap2] [Chapter 2][cap2] in the book "Dive into Deep Learning." Other topics like information theory or the maximum likelihood principle are covered in the "Information Theory" and "Maximum Likelihood" sections of an [:octicons-book-24:][appendix] [appendix][appendix] in the same book.

[cap2]: https://d2l.ai/chapter_preliminaries/index.html
[appendix]: https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html

## Further Reading

Expand your knowledge with the following books, most of which are available online:

- "[Speech and Language Processing][jurafskybib]" by Dan Jurafsky and James H. Martin. Third edition unpublished as of 2024 but with an advanced draft online. Details key NLP concepts and models without delving into implementation details. This guide is based on this book.
- "[Dive into Deep Learning][d2l]" by Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola. Explores deep learning models in detail, with a [paper version][cambridge] published in 2023.
- ["Deep Learning: Foundations and Concepts"][bishop] by Chris Bishop and Hugh Bishop. Also available in print since 2024.
- "[Understanding Deep Learning][understanding]" (2023) by Simon J.D. Prince. Filled with illustrations and figures to clarify concepts.
- The series "[Probabilistic Machine Learning: An Introduction][pml1]" (2022) and "[Probabilistic Machine Learning: Advanced Topics][pml2]" (2023) by Kevin Murphy covers various machine learning elements in depth.
- "[Deep Learning for Natural Language Processing: A Gentle Introduction][gentle]" by Mihai Surdeanu and Marco A. Valenzuela-Escárcega. Still under development. Contains code in some chapters.
- "[Deep Learning with PyTorch Step-by-Step: A Beginner's Guide][pytorchstep]" (2022) by Daniel Voigt Godoy. Paid, with digital and print versions (in three volumes). A Spanish version of the first chapters exists. Written in a clear, example-rich style.
- "[The Mathematical Engineering of Deep Learning][mathdl]" (2024) by Benoit Liquet, Sarat Moka, and Yoni Nazarathy.
- "[The Little Book of Deep Learning][little]" (2023) by François Fleuret.

[jurafskybib]: https://web.stanford.edu/~jurafsky/slp3/
[gentle]: https://clulab.org/gentlenlp/
[understanding]: https://udlbook.github.io/udlbook/
[pytorchstep]: https://leanpub.com/pytorch
[bishop]: https://www.bishopbook.com/
[d2l]: http://d2l.ai/
[pml1]: https://probml.github.io/pml-book/book1.html
[pml2]: https://probml.github.io/pml-book/book2.html
[cambridge]: https://www.cambridge.org/es/academic/subjects/computer-science/pattern-recognition-and-machine-learning/dive-deep-learning?format=PB
[mathdl]: https://deeplearningmath.org/
[little]: https://fleuret.org/francois/lbdl.html

The following list includes links to video courses by renowned researchers or universities:

- [Stanford CS224n][cs224n] ― Natural Language Processing with Deep Learning; [course website][cs224nweb].
- [Stanford CS324][cs324] ― Large Language Models; 2022 edition.
- [Stanford CS324 2023][cs324b] ― Advances in Foundation Models; 2023 edition.
- [Stanford CS25][cs25] ― Transformers United; [course website][cs25web].
- [Stanford CS229][cs229] ― Machine Learning; [course website][cs229web].
- [Stanford CS230][cs230] ― Deep Learning; [course website][cs230web].
- [MIT 6.S191][mit191] ― Introduction to Deep Learning; [course website][mit191web].
- "Neural Networks: [Zero to Hero][zero2hero]" by Andrew Karpathy.
- [Machine Learning][mlspec] Specialization.

[mlspec]: https://www.youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI
[cs229]: https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU
[cs229web]: https://cs229.stanford.edu/
[cs230]: https://youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb
[cs230web]: https://cs230.stanford.edu/
[cs224n]: https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
[cs224nweb]: http://web.stanford.edu/class/cs224n/
[cs324]: https://stanford-cs324.github.io/winter2022/
[cs324b]: https://stanford-cs324.github.io/winter2023/
[cs25]: https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM
[cs25web]: https://web.stanford.edu/class/cs25/
[mit191]: https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI
[mit191web]: http://introtodeeplearning.com
[zero2hero]: https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
