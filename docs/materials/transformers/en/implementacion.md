
<div class="content-2columns" markdown>
![](../assets/imgs/implementation-heart.png){: .rounded-title-img}

# Implementation of Models in PyTorch
</div>

The implementation of different models in code is a complementary approach to studying them from a mathematical perspective. This page presents PyTorch implementations of each of the studied models. The idea is to approach these implementations after conceptually studying the respective model.

{%
   include-markdown "./assets/mds/texts.md"
   start="<!--nota-inicial-start-->"
   end="<!--nota-inicial-end-->"
%}

## Code for a Logistic and a Multinomial Regressor {#code-regressor}

Here are two PyTorch implementations of the regressors studied [on this page](regresor.md) in just a few dozen lines of code. Ensure you understand the code well enough to feel confident about modifying it to suit other needs.

Review how to [debug](pytorch.md#debug) Python programs before tackling the code. Also, review how the [broadcasting](apuntes.md#broadcasting) mechanism works in PyTorch.

The two programs in this section are:

- A [:octicons-book-24:][pylog] [logistic regressor][pylog] that classifies two-dimensional synthetic samples into two classes. Only the most basic elements of PyTorch are used to keep the implementation as detailed as possible. As an exercise, trace and analyze the sizes of the tensors. Experiment with the number of training steps and the learning rate to observe how training evolves. Explore various positions of class centers and data dispersion to see how the decision boundary changes. Remove the bias from the equations and observe how forcing the decision boundary to pass through the origin restricts its shape. <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/logistic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- A [:octicons-book-24:][pysoft] [softmax regressor for classifying texts by topic][pysoft]. As an exercise, try training with a single embedding per training step instead of a batch of all embeddings and observe how the error behaves. You can also adapt the previous logistic regressor code to use the PyTorch functions seen in this program. <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/softmax.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you haven't already, you can start learning Python and PyTorch by following the [chapter][cappy] corresponding to this series.

[cappy]: pytorch.md
[pylog]: ../assets/notebooks/logistic.ipynb
[pysoft]: ../assets/notebooks/softmax.ipynb

## Code for Skip-Grams {#code-skipgrams}

This is an implementation of the [:octicons-book-24:][pyskip] [skip-gram][pyskip] algorithm for obtaining static embeddings as studied [on this page](embeddings.md). It follows the guidelines in the book by Jurafsky and Martin. <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/skipgram.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

[pyskip]: ../assets/notebooks/skipgram.ipynb

## Code for a Language Model with Feedforward Networks {#code-ffw}

This is the implementation of a [:octicons-book-24:][pylm] [language model][pylm] using feedforward networks as studied [on this page](ffw.md). It adheres to the equations in the book by Jurafsky and Martin. <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/ffnn.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

[pylm]: ../assets/notebooks/ffnn.ipynb

## Code for the Transformer {#code-transformer}

The transformer (studied in this [section](attention.md) of the guide) is presented in three separate notebooks:

- One that contains the [:octicons-book-24:][pytr] [base architecture][pytr] and implementations for both an encoder-based model and a decoder-based model. <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/transformer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- Another that applies the decoder to a [:octicons-book-24:][pygpt] [language model][pygpt] predicting the next token in a sequence. <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/lmgpt.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- And one based on the encoder to build a [:octicons-book-24:][pyner] [named entity recognition system][pyner]. <a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/transformers/assets/notebooks/nerbert.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

[pytr]: ../assets/notebooks/transformer.ipynb
[pygpt]: ../assets/notebooks/lmgpt.ipynb
[pyner]: ../assets/notebooks/nerbert.ipynb

## Code for a Transformer from the minGPT Project {#code-mingpt}

A good PyTorch implementation of a transformer-based language model is Andrej Karpathy's [minGPT][minGPT]. The code allows for training and using language models and loading the weights of the GPT-2 model. The transformer in our guide is based on minGPT, so the model itself should not be difficult to understand.

This guide's repository has a [copy][copia] of the minGPT code with minor modifications. Below is a summary of relevant files. You do not need to examine files that are not mentioned. To use and modify the code, you can install it with:

```bash
pip install --editable .
```

[minGPT]: https://github.com/karpathy/minGPT
[vidkarpathy]: https://youtu.be/kCc8FmEb1nY
[copia]: ../assets/code/minGPT-20230108/README.md

Due to changes in external dependencies, the current code may not work as-is. To fix this, modify line 200 of the file `mingpt/model.py` from:

```python
assert len(keys) == len(sd)
```

to:

```python
assert len(keys) == len([k for k in sd if not k.endswith(".attn.bias")])
```

### File mingpt/bpe.py

This file contains the necessary implementation to use the BPE subword model used by GPT-2. Its functionality is discussed later. The main code in the file demonstrates a step-by-step tokenization example of an input string, which you can see by running `python bpe.py`. The first time the `encode` or `decode` methods are called, the files `encoder.json` and `vocab.bpe`—containing the vocabulary and subword merge rules used by GPT-2, respectively—are downloaded. These files are stored in the `~/.cache/mingpt` directory.

It is not necessary to study the code in this file. Simply know that it allows you to obtain a list of token indices from an input text and retrieve the text associated with a list of token indices output by the model:

```python
bpe = BPETokenizer()
tokens = bpe("A relaxing cup of café con leche in Plaza Mayor") # encode
# tokens is a tensor of shape (1, 9)
print(bpe.decode(tokens[0]))  
# "A relaxing cup of café con leche in Plaza Mayor"
print(tokens[0].tolist()) 
# [32, 28175, 6508, 286, 40304, 369, 443, 2395, 287, 23280, 10106]
for token in tokens[0]:
   print(bpe.decode(torch.tensor([token])), end='/')
# A/ relaxing/ cup/ of/ café/ con/ le/che/ in/ Plaza/ Mayor/
```

### File mingpt/utils.py

It is not necessary to study this file in detail. Simply open it to observe that it defines two utility functions (`set_seed` and `setup_logging`) and a class (`CfgNode`) for managing the model's configuration parameters.

### File mingpt/trainer.py

Study this file, as it contains the general code responsible for training a model. The code is not specific to the transformer architecture and could be applied with minor modifications to other models.

### File mingpt/model.py

The most important file for our purposes. However, you can skip the `from_pretrained` method of the `GPT` class (incorporates GPT-2 weights downloaded from Hugging Face Transformers) and especially the `configure_optimizers` method (returns an Adam optimizer with different behavior depending on the type of parameter it acts upon), as they contain code specific to the GPT-2 system.

Study the `CausalSelfAttention` and `Block` classes in detail, as well as the `forward`, `generate`, `__init__`, `_init_weights`, and `get_default_config` methods of the `GPT` class.

### File generate.ipynb

Study this code, which uses the model to generate text. It is a Python notebook but can be executed from the command line by converting it to a Python script:

```bash
pip install nbconvert
jupyter nbconvert --to script generate.ipynb
python generate.py
```

You can change the `model-type` variable to use different pre-trained GPT-2 models. From largest to smallest, the available models are `gpt2-xl`, `gpt2-large`, `gpt2-medium`, and `gpt2`. If you want to run the code on a CPU, change the `device` value to:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### File projects/chargpt/charpgt.py

This code trains a character-level language model using the content of the `input.txt` file. You can use texts such as [Don Quixote][quijote] or parts of [Shakespeare's works][shakespeare] as input files.

[quijote]: https://www.gutenberg.org/cache/epub/2000/pg2000.txt
[shakespeare]: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

You can change the `C.model.model_type` variable to use models of different sizes (from largest to smallest: `gpt2-xl`, `gpt2-large`, `gpt2-medium`, `gpt2`, `gpt-mini`, `gpt-micro`, and `gpt-nano`). The number of layers, attention heads, and embedding sizes for each model can be found in the `GPT` class constructor in the `mingpt/model.py` file.

Run the program and let it train for a while with:

```bash
python charpgt.py
```

The model is saved periodically in the `out` folder.

## Additional Implementations

The [MinT][MinT] project includes various tutorials with scratch implementations of models like BERT, GPT, BART, or T5. The code is slightly more extensive than what we have studied but can help consolidate knowledge at an advanced stage. The [x-transformers][x-transformers] project follows a similar approach.

There is some competition among developers to achieve the most compact transformer implementation possible. Some notable ones are [minGPT][mingpt], [nanoGPT][nanogpt], and [picoGPT][picogpt]. A notable feature of these implementations is their ability to load GPT-2 weights and perform inference. Andrej Karpathy, the developer of minGPT and nanoGPT, has a highly educational [video][video] explaining his implementation.

[MinT]: https://github.com/dpressel/mint
[x-transformers]: https://github.com/lucidrains/x-transformers

[mingpt]: https://github.com/karpathy/minGPT
[nanogpt]: https://github.com/karpathy/nanoGPT
[picogpt]: https://github.com/jaymody/picoGPT
[video]: https://youtu.be/kCc8FmEb1nY

